import base64
import json
import os
import threading

import requests
from tqdm import tqdm

# Load IPD_PASSWORD (and anything else) from scripts/.env if present
def _load_dotenv():
  env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
  if not os.path.exists(env_path):
    return
  with open(env_path) as f:
    for line in f:
      line = line.strip()
      if not line or line.startswith("#") or "=" not in line:
        continue
      key, _, value = line.partition("=")
      value = value.strip()
      if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
      os.environ.setdefault(key.strip(), value)


_load_dotenv()

# 1. Base configuration
BASE_URL = "https://india-data.org"
DOWNLOAD_URL = f"{BASE_URL}/du/download/v1/download-file"
RENEW_URL = f"{BASE_URL}/auth/renew-token"
LOGIN_URL = f"{BASE_URL}/auth/login"

# 2. Session tokens. These are live bearer credentials, so none are
# hardcoded here -- they're loaded from the environment (via scripts/.env,
# which is gitignored and never committed). Leaving them unset is the
# normal/expected case on a fresh checkout: the script falls back to a full
# username+password login (see _login_with_password) to mint a new set.
refresh_token = os.environ.get("IPD_REFRESH_TOKEN", "")
session_id = os.environ.get("IPD_SESSION_ID", "")
access_token = os.environ.get("IPD_ACCESS_TOKEN", "")

file_name_path = "170acc68-1288-499e-9a91-b951e569e70d/ac676a96-1f94-4c52-a83a-2eb204f71c91/DATASET-FILE/Labeled_Part_4_zip_20241226073023476.zip"
output_filename = "Labeled_Part_4.zip"
state_filename = output_filename + ".progress.json"

# Server honours Range requests (that's how the old single-stream resume
# worked), so splitting the file into N byte ranges and pulling them over
# separate connections gets around the single-TCP-stream throughput cap.
NUM_CONNECTIONS = 8
CHUNK_READ_SIZE = 1024 * 1024  # 1MB read size per request

headers = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Encoding": "identity",
    "app-name": "DFS",
    "fileName": file_name_path,
    "Referer": "https://india-data.org/dataset-details/170acc68-1288-499e-9a91-b951e569e70d",
}


def get_cookie_string(acc_tok, ref_tok, ses_id):
  return (
      f"access-token-dfs={acc_tok}; refresh-token-dfs={ref_tok};"
      f" session-id={ses_id}; myAwesomeCookieName2=true"
  )


def build_headers():
  h = headers.copy()
  h["Cookie"] = get_cookie_string(access_token, refresh_token, session_id)
  return h


def decode_jwt_payload(token):
  payload_b64 = token.split(".")[1]
  padded = payload_b64 + "=" * (-len(payload_b64) % 4)
  return json.loads(base64.urlsafe_b64decode(padded))


token_lock = threading.Lock()


def _renew_with_refresh_token():
  """POST to the renew-token endpoint (discovered from the site's frontend
  bundle) to mint a fresh access token off the still-valid refresh-token
  cookie. Returns True on success. Caller must hold token_lock."""
  global access_token
  user_id = decode_jwt_payload(refresh_token)["userid"]
  r = requests.post(
      RENEW_URL,
      json={"userId": user_id, "appName": "dfs"},
      headers={
          "Session-Id": session_id,
          "app-name": "dfs",
          "Cookie": get_cookie_string(access_token, refresh_token, session_id),
      },
      timeout=30,
  )
  new_token = r.headers.get("access-token-dfs")
  if r.status_code == 201 and new_token:
    access_token = new_token
    return True
  return False


def _login_with_password():
  """POST to the login endpoint (also found in the frontend bundle) to get
  a whole new access/refresh/session triple. Used on a fresh checkout (no
  tokens set at all) and again once the refresh token itself has expired
  (roughly hourly) -- the renew-token endpoint alone can't recover from
  that, only a real login can. Requires IPD_USERNAME and IPD_PASSWORD to be
  set in the environment; we never hardcode credentials in this file.
  Caller must hold token_lock."""
  global access_token, refresh_token, session_id
  username = os.environ.get("IPD_USERNAME")
  password = os.environ.get("IPD_PASSWORD")
  if not username or not password:
    print(
        "\nNo valid session and IPD_USERNAME/IPD_PASSWORD are not set in the"
        " environment -- can't auto-login. Set both (e.g. in scripts/.env)"
        " and restart."
    )
    return False
  r = requests.post(
      LOGIN_URL,
      json={"userName": username, "password": password, "appName": "dfs"},
      timeout=30,
  )
  new_access = r.headers.get("access-token-dfs")
  new_refresh = r.headers.get("refresh-token-dfs")
  new_session = r.headers.get("session-id")
  if r.status_code in (200, 201) and new_access and new_refresh:
    access_token, refresh_token = new_access, new_refresh
    if new_session:
      session_id = new_session
    print("\nLogged in fresh (refresh token had expired).")
    return True
  print(f"\nAuto-login failed: HTTP {r.status_code} {r.text[:200]}")
  return False


def reauth(stale_token):
  """Re-authenticate after a 401/403/498. `stale_token` is the access_token
  the caller saw as expired -- if another thread already fixed it while we
  waited for the lock, skip redundant network calls. Tries the cheap
  refresh-token renewal first, and falls back to a full password login if
  the refresh token itself has expired (or was never set at all)."""
  with token_lock:
    if access_token != stale_token:
      return True  # a different thread already re-authenticated
    if refresh_token and _renew_with_refresh_token():
      return True
    return _login_with_password()


def get_total_size():
  """Probe the file size with a 1-byte range request."""
  for _ in range(5):
    stale = access_token
    probe_headers = build_headers()
    probe_headers["Range"] = "bytes=0-0"
    with requests.get(
        DOWNLOAD_URL, headers=probe_headers, stream=True, timeout=30
    ) as r:
      if r.status_code in (401, 403, 498):
        if reauth(stale):
          continue
        print(
            "Error: re-authentication failed (both renew-token and login)."
            " Set/fix IPD_USERNAME and IPD_PASSWORD in the environment and"
            " re-run."
        )
        exit(1)
      r.raise_for_status()
      if r.status_code == 206 and "Content-Range" in r.headers:
        return int(r.headers["Content-Range"].split("/")[-1])
      # Server didn't honour the range request -- fall back to Content-Length.
      return int(r.headers.get("Content-Length", 0))
  raise RuntimeError("Could not determine file size after repeated renewals")


def make_chunks(total_size, num_connections):
  chunk_size = total_size // num_connections
  chunks = []
  start = 0
  for i in range(num_connections):
    end = total_size - 1 if i == num_connections - 1 else start + chunk_size - 1
    chunks.append([start, end])
    start = end + 1
  return chunks


def load_state(total_size, chunks):
  if os.path.exists(state_filename):
    with open(state_filename) as f:
      state = json.load(f)
    if state.get("total_size") == total_size and len(state.get("done", [])) == len(chunks):
      return state["done"]
  return [0] * len(chunks)


state_lock = threading.Lock()


def save_state(total_size, done):
  with state_lock:
    with open(state_filename, "w") as f:
      json.dump({"total_size": total_size, "done": done}, f)


def download_chunk(idx, start, end, done, pbar, stop_event, total_size):
  # The server 416s on closed ranges ("bytes=start-end") and, even with an
  # open-ended range ("bytes=start-"), only ever returns a capped amount per
  # request rather than streaming to EOF -- so each chunk has to be fetched
  # as a series of open-ended requests, advancing start each time, and any
  # response that overshoots our chunk's end is truncated client-side.
  try:
    while True:
      range_start = start + done[idx]
      if range_start > end:
        return  # chunk fully downloaded

      stale = access_token
      chunk_headers = build_headers()
      chunk_headers["Range"] = f"bytes={range_start}-"

      with requests.get(
          DOWNLOAD_URL, headers=chunk_headers, stream=True, timeout=30
      ) as r:
        if r.status_code in (401, 403, 498):
          if reauth(stale):
            continue  # retry this same range with the fresh token
          print(
              f"\nChunk {idx}: re-authentication failed (both renew-token and"
              " login). Set/fix IPD_USERNAME and IPD_PASSWORD in the"
              " environment and restart (already-downloaded chunks will"
              " resume)."
          )
          stop_event.set()
          return
        r.raise_for_status()

        got_any = False
        with open(output_filename, "r+b") as f:
          f.seek(range_start)
          for piece in r.iter_content(chunk_size=CHUNK_READ_SIZE):
            if stop_event.is_set():
              return
            if not piece:
              continue
            got_any = True
            remaining = end - (start + done[idx]) + 1
            if len(piece) > remaining:
              piece = piece[:remaining]
            f.write(piece)
            done[idx] += len(piece)
            with state_lock:
              pbar.update(len(piece))
            save_state(total_size, done)
            if start + done[idx] > end:
              return  # chunk complete, stop reading even if server has more

        if not got_any:
          print(f"\nChunk {idx} stalled (empty response). Re-run to resume.")
          stop_event.set()
          return
  except Exception as e:
    print(f"\nChunk {idx} failed: {e}. Re-run the script to resume it.")
    stop_event.set()


def main():
  total_size = get_total_size()
  chunks = make_chunks(total_size, NUM_CONNECTIONS)
  done = load_state(total_size, chunks)

  # Pre-allocate (or reuse) the output file at full size so each thread
  # can seek+write its own byte range independently.
  if not os.path.exists(output_filename) or os.path.getsize(output_filename) != total_size:
    with open(output_filename, "wb") as f:
      f.truncate(total_size)

  already_done = sum(done)
  if already_done >= total_size:
    print(f"{output_filename} is already fully downloaded.")
    return

  print(
      f"Downloading {output_filename} ({total_size / (1024**2):.1f} MB) "
      f"using {NUM_CONNECTIONS} connections, resuming from "
      f"{already_done / (1024**2):.1f} MB"
  )

  stop_event = threading.Event()
  with tqdm(
      total=total_size,
      initial=already_done,
      unit="B",
      unit_scale=True,
      unit_divisor=1024,
      desc=output_filename,
  ) as pbar:
    threads = []
    for idx, (start, end) in enumerate(chunks):
      t = threading.Thread(
          target=download_chunk,
          args=(idx, start, end, done, pbar, stop_event, total_size),
      )
      t.start()
      threads.append(t)
    for t in threads:
      t.join()

  if stop_event.is_set():
    print("\nDownload incomplete. Re-run the script anytime to resume.")
    exit(1)

  os.remove(state_filename)
  print("\nDownload completed successfully!")


if __name__ == "__main__":
  main()
