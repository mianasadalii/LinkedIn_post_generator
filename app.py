import os
import gradio as gr
from groq import Groq

# ─── Groq Client (reads from HF Secret) ─────────────────────────────────────

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

def get_client() -> Groq:
    return Groq(api_key=GROQ_API_KEY)


# ─── Post Generation Logic ───────────────────────────────────────────────────

TONES = ["Professional", "Inspirational", "Storytelling", "Casual & Friendly", "Bold & Controversial", "Educational"]
POST_TYPES = ["Achievement / Win", "Career Lesson", "Industry Insight", "Product / Service Promo", "Personal Story", "Thought Leadership", "Tips & Advice"]
LENGTHS = ["Short (≤150 words)", "Medium (150–300 words)", "Long (300–500 words)"]

SYSTEM_PROMPT = """You are a world-class LinkedIn content strategist and copywriter.
Your job is to write high-performing LinkedIn posts that:
- Hook readers in the first line (no fluff openers)
- Use short paragraphs and white space for mobile readability
- Include a clear value proposition or lesson
- End with an engaging call-to-action or thought-provoking question
- Use relevant emojis sparingly but effectively
- Feel authentic, not corporate or robotic
Return ONLY the post text, ready to copy-paste. No explanations, no labels."""

def build_user_prompt(topic, post_type, tone, length, keywords, include_hashtags, include_cta):
    length_guide = {
        "Short (≤150 words)": "Keep the post under 150 words. Be punchy and direct.",
        "Medium (150–300 words)": "Aim for 150–300 words. Balance depth with readability.",
        "Long (300–500 words)": "Write 300–500 words. Tell a fuller story with substance.",
    }
    prompt = f"""Write a LinkedIn post with the following specs:
Topic: {topic}
Post Type: {post_type}
Tone: {tone}
Length: {length_guide[length]}"""

    if keywords.strip():
        prompt += f"\nKeywords to naturally include: {keywords}"
    if include_hashtags:
        prompt += "\nAdd 3–5 relevant hashtags at the end."
    if include_cta:
        prompt += "\nEnd with a strong call-to-action that encourages engagement (comment, share, or connect)."
    return prompt


def generate_post(topic, post_type, tone, length, keywords, include_hashtags, include_cta, model):
    if not topic.strip():
        return "⚠️ Please enter a topic for your post."
    try:
        client = get_client()
        user_prompt = build_user_prompt(topic, post_type, tone, length, keywords, include_hashtags, include_cta)
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
            temperature=0.85,
            max_tokens=1024,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error: {str(e)}"


def regenerate_post(topic, post_type, tone, length, keywords, include_hashtags, include_cta, model):
    return generate_post(topic, post_type, tone, length, keywords, include_hashtags, include_cta, model)


# ─── Gradio UI ───────────────────────────────────────────────────────────────

CSS = """
:root {
    --linkedin-blue: #0A66C2;
    --linkedin-dark: #004182;
    --surface: #F3F6F9;
    --border: #D8E0E8;
    --text: #1B1F23;
    --muted: #5E6D7A;
}
body, .gradio-container {
    font-family: 'Segoe UI', system-ui, sans-serif !important;
    background: var(--surface) !important;
    color: var(--text) !important;
}
.app-header {
    background: linear-gradient(135deg, var(--linkedin-blue) 0%, var(--linkedin-dark) 100%);
    color: white;
    padding: 28px 32px 24px;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(10,102,194,0.25);
}
.app-header h1 { margin: 0 0 4px; font-size: 26px; font-weight: 700; }
.app-header p  { margin: 0; opacity: 0.85; font-size: 14px; }
.section-label {
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    margin-bottom: 12px;
}
.output-box textarea {
    font-family: 'Segoe UI', sans-serif !important;
    font-size: 15px !important;
    line-height: 1.65 !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 8px !important;
    background: #FAFCFF !important;
    padding: 16px !important;
    min-height: 260px !important;
}
.tip-box {
    background: #EBF5FB;
    border-left: 4px solid var(--linkedin-blue);
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    font-size: 13px;
    color: #1a3a5c;
    margin-top: 8px;
}
"""

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

with gr.Blocks(css=CSS, title="LinkedIn Post Generator") as demo:

    gr.HTML("""
    <div class="app-header">
        <h1>🔵 LinkedIn Post Generator</h1>
        <p>Powered by Groq's ultra-fast inference — craft scroll-stopping LinkedIn posts in seconds.</p>
    </div>
    """)

    with gr.Row(equal_height=False):

        # ── Left Column: Inputs ──
        with gr.Column(scale=1):

            gr.HTML('<div class="section-label">⚙️ Model</div>')
            model = gr.Dropdown(
                label="Groq Model",
                choices=GROQ_MODELS,
                value=GROQ_MODELS[0],
            )

            gr.HTML('<br><div class="section-label">✍️ Post Details</div>')

            topic = gr.Textbox(
                label="Topic / Main Idea",
                placeholder="e.g. I just got promoted after 3 years of hard work...",
                lines=3,
            )
            post_type = gr.Dropdown(
                label="Post Type",
                choices=POST_TYPES,
                value=POST_TYPES[0],
            )
            tone = gr.Dropdown(
                label="Tone",
                choices=TONES,
                value=TONES[0],
            )
            length = gr.Radio(
                label="Post Length",
                choices=LENGTHS,
                value=LENGTHS[1],
            )
            keywords = gr.Textbox(
                label="Keywords to Include (optional)",
                placeholder="e.g. leadership, growth mindset, remote work",
            )
            with gr.Row():
                include_hashtags = gr.Checkbox(label="Add Hashtags", value=True)
                include_cta = gr.Checkbox(label="Add Call-to-Action", value=True)

            gr.HTML("""
            <div class="tip-box">
                💡 <strong>Tip:</strong> Be specific in your topic! Instead of "leadership tips",
                try "3 things I learned about leadership after managing my first team of 10 people."
            </div>
            """)

        # ── Right Column: Output ──
        with gr.Column(scale=1):

            gr.HTML('<div class="section-label">📄 Generated Post</div>')

            output = gr.Textbox(
                label="",
                placeholder="Your LinkedIn post will appear here...",
                lines=16,
                show_copy_button=True,   # works in Gradio 5
                elem_classes=["output-box"],
            )

            with gr.Row():
                generate_btn = gr.Button("✨ Generate Post", variant="primary")
                regen_btn    = gr.Button("🔄 Regenerate", variant="secondary")

            gr.HTML("""
            <div class="tip-box" style="margin-top:12px">
                📋 <strong>After generating:</strong> Use the copy icon on the text box,
                then paste directly into LinkedIn. Always add a photo or image to boost reach by 2–3×.
            </div>
            """)

            gr.HTML('<br><div class="section-label">🚀 Quick-start Examples</div>')
            gr.Examples(
                examples=[
                    ["I just completed my first open-source project after 6 months of learning to code on weekends.", "Achievement / Win", "Inspirational", "Medium (150–300 words)", "open source, side project, consistency", True, True],
                    ["Why most people fail at networking — and the one mindset shift that changed everything for me.", "Thought Leadership", "Bold & Controversial", "Medium (150–300 words)", "networking, career growth, LinkedIn", True, True],
                    ["5 Python tricks I wish I knew when I started data science 2 years ago.", "Tips & Advice", "Educational", "Long (300–500 words)", "Python, data science, productivity", True, False],
                ],
                inputs=[topic, post_type, tone, length, keywords, include_hashtags, include_cta],
                label="",
            )

    # ── Event Handlers ──
    inputs = [topic, post_type, tone, length, keywords, include_hashtags, include_cta, model]
    generate_btn.click(fn=generate_post, inputs=inputs, outputs=output)
    regen_btn.click(fn=regenerate_post, inputs=inputs, outputs=output)


# ─── Launch ──────────────────────────────────────────────────────────────────

demo.launch()
