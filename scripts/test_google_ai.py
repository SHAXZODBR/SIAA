"""Quick test: which Gemini models work with your API key."""
import os
import sys

api_key = os.environ.get('GOOGLE_AI_KEY')
if not api_key:
    print("ERROR: GOOGLE_AI_KEY not set")
    sys.exit(1)

try:
    import google.generativeai as genai
    genai.configure(api_key=api_key)

    print("\n" + "=" * 60)
    print("MODELS AVAILABLE WITH YOUR KEY:")
    print("=" * 60)

    available = []
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            available.append(m.name)
            print(f"  ✓ {m.name}")

    if not available:
        print("  No models support generateContent!")
        sys.exit(1)

    # Test the first available model
    print("\n" + "=" * 60)
    print(f"TESTING: {available[0]}")
    print("=" * 60)

    model = genai.GenerativeModel(available[0])
    response = model.generate_content("Say hello in Russian, Uzbek, and English.")
    print(response.text)
    print("\n✓ SUCCESS! Use this model name in CONFIG:", available[0])

except ImportError:
    print("Install: pip install google-generativeai")
except Exception as e:
    print(f"ERROR: {e}")
