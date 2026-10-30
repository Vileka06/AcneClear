# Connect the AcneClear AI chatbot using a free API tier

This setup does not require paid GPU hosting, Hugging Face Spaces, Modal, or uploading the Qwen LoRA adapter.

The live Streamlit chatbot, Beauty Tips and Diet Planner use Google's Gemini API Free Tier. Your Qwen3-0.6B LoRA adapter remains a separate trained Colab experiment.

## 1. Create a free API key

1. Open https://aistudio.google.com/apikey
2. Sign in with your Google account.
3. Create an API key.
4. Keep the key private.
5. Do not enable paid billing or connect a paid billing account.

The app calls the stable model ID `gemini-3.5-flash-lite`. Google currently lists free input and output tokens for this model on the Gemini API Free Tier. Free access is limited by account/project eligibility, model availability and rate limits. Check current quotas in Google AI Studio.

Official references:
- Pricing: https://ai.google.dev/gemini-api/docs/pricing
- Model: https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite
- Rate limits: https://ai.google.dev/gemini-api/docs/rate-limits
- Create API key: https://aistudio.google.com/apikey

**Privacy note:** Google lists free-tier content as potentially used to improve its products. Do not send private photos, passwords, sensitive medical history, or other confidential information to the chatbot.

## 2. Add the API key to Streamlit Community Cloud

1. Open https://share.streamlit.io/
2. Select the AcneClear app.
3. Open the app menu and choose **Settings**.
4. Open **Secrets**.
5. Paste:

```toml
GEMINI_API_KEY = "paste-your-google-ai-studio-key-here"
```

6. Save the secrets. Streamlit will restart the application.

Do not put the key in app.py, GitHub, a screenshot, or a public file. The app sends API requests from the server; the key is not sent to the browser. Streamlit's official secrets documentation: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management

## 3. Test the live application

Open https://acneclear.streamlit.app/ and test each module:

### AI Chatbot
- "Explain my complete AcneClear AI project architecture."
- "What is the difference between GFPGAN and Qwen3?"
- "Explain LoRA and my three training epochs."
- "What were my test loss and perplexity?"
- "Explain Python functions with an example."

### Beauty Tips
Select a skin type and concern. Ask for a simple morning and evening skincare routine.

### Diet Planner
Choose food preference, meal style and budget. Ask for general Indian meal ideas.

### Acne Remover
Upload a test portrait and confirm the existing GFPGAN flow still works.

## 4. How to explain the model choice to faculty

"I fine-tuned Qwen3-0.6B using LoRA in Google Colab. To keep the live website available without paying for dedicated GPU hosting, I connected the chatbot, Beauty Tips and Diet Planner to Gemini's free API tier. The Qwen3 LoRA adapter is retained as my separate fine-tuning experiment; the deployed chatbot currently uses Gemini."

Do not say the live site is running Qwen3 LoRA. It is not in this free setup.

## 5. Troubleshooting

- **Chatbot says API key is missing:** Check that the secret is named exactly `GEMINI_API_KEY`.
- **403 error:** Check that the API key is valid and the Google AI Studio project is eligible for the free tier. Do not enable billing if you want zero cost.
- **429 error:** The free-tier rate limit has been reached. Wait for the limit to reset and check Google AI Studio quotas.
- **API model not found:** Verify that `gemini-3.5-flash-lite` remains available to your project in AI Studio.
- **App does not rebuild:** Check Streamlit Community Cloud logs.

## 6. GFPGAN remains separate

The existing GFPGAN v1.3 photo-restoration code and OpenCV skin-retouching functions are kept unchanged. The Gemini API handles text only; it does not replace the photo-restoration model.
