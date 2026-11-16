# AcneClear mobile app (Mac)

The mobile starter uses Expo SDK 52. The website and API can be run first with the root-level `START_ACNECLEAR.command`.

## Install and start

1. Install Node.js LTS from https://nodejs.org if it is not installed.
2. Open Terminal in the AcneClear folder.
3. Run:

```bash
cd mobile
npm config set fetch-retries 5
npm config set fetch-timeout 120000
npm install
npx expo start
```

If npm asks to install a different Expo version, answer **N**. The project already declares Expo SDK 52.

## Connect a real phone to the API

Find your Mac's local Wi-Fi IP address in System Settings > Wi-Fi > Details, then edit the `API_URL` near the top of `App.js` to:

```js
const API_URL = "http://YOUR-MAC-LAN-IP:8000";
```

Keep the phone and Mac on the same Wi-Fi network. The API must be running. For Android emulator use `http://10.0.2.2:8000`; for iOS simulator use `http://localhost:8000`.

If npm reports ETIMEDOUT, that is a network/registry connection issue, not an application-code error. Retry on a stable network. Do not install a different Expo major version to work around it.
