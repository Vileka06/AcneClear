# AcneClear AI mobile app

The Expo React Native app has one capability: upload a portrait to the FastAPI service, enhance it with GFPGAN, and preview the result.

## Install and start

1. Install Node.js LTS from https://nodejs.org if it is not installed.
2. Open Terminal in the AcneClear folder.
3. Run:

~~~bash
cd mobile
npm config set fetch-retries 5
npm config set fetch-timeout 120000
npm install
~~~

Set the API URL before starting Expo:

~~~bash
export EXPO_PUBLIC_API_URL=http://YOUR-MAC-LAN-IP:8000
npx expo start
~~~

For a physical phone, use your Mac's local Wi-Fi IP address instead of localhost. Keep the phone and Mac on the same Wi-Fi network. The API must be running and reachable from the phone. For Android emulator use http://10.0.2.2:8000; for iOS simulator use http://localhost:8000.

If npm reports ETIMEDOUT, that is a network/registry connection issue. Retry on a stable network.
