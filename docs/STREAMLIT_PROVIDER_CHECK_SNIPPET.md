Provider Health Check Snippets

1) Google Generative Language (quick probe)

curl -s -X GET "https://generativelanguage.googleapis.com/v1beta/models?key=$GOOGLE_API_KEY" | jq .

2) Google generation test

curl -s -X POST "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=$GOOGLE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"contents":[{"parts":[{"text":"اجب بالعربية: ما حكم التلبس؟"}]}], "generationConfig":{"maxOutputTokens":80,"temperature":0.2}}' | jq .

3) OpenAI model list probe

curl -s -H "Authorization: Bearer $OPENAI_API_KEY" https://api.openai.com/v1/models | jq .

Note: use short timeouts in production health checks (6-10s) and surface HTTP code + error body for debugging.