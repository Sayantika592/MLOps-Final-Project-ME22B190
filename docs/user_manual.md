# User Manual — Fake News Detection System

## Welcome

The Fake News Detection System helps you check whether a news article is likely **real** or **fake**. Simply paste the article text and click a button to get a result.

---

## Getting Started

### Accessing the Application

1. Open your web browser (Chrome, Firefox, Edge, or Safari).
2. Navigate to: **http://localhost:8501**
3. The application loads with the **Prediction** page visible by default.

### System Requirements

- A modern web browser with JavaScript enabled.
- The system must be running (see the "System Status" indicator in the sidebar).
  - **✅ API Online** means the system is ready.
  - **❌ API Offline** means the backend is not running — contact the administrator.

---

## How to Check a News Article

### Step 1: Enter the Text

On the **🔍 Prediction** page, you will see a large text box. You can either:

- **Type** or **paste** a news headline or full article into the text box.
- Click one of the **Quick Example** buttons on the right to try a sample.

### Step 2: Click "Analyze"

Click the blue **🔎 Analyze** button below the text box.

The system will process your text (usually takes 1–2 seconds).

### Step 3: Read the Result

After analysis, you will see:

- A colored banner:
  - **Green** = The article appears to be **REAL** news ✅
  - **Red** = The article appears to be **FAKE** news 🚨
- **Confidence Score**: How certain the model is (higher = more certain).
- **Response Time**: How long the analysis took.
- **Word Count**: Number of words in your text.

A **gauge chart** also shows the confidence visually.

---

## Navigation Guide

Use the sidebar on the left to switch between pages:

| Page | What It Shows |
|------|---------------|
| 🔍 Prediction | The main analysis tool |
| 📊 Pipeline Dashboard | Technical details about how the model was built |
| 🛡️ Monitoring | System health and data drift status |
| 📖 User Manual | This help page |

---

## Tips for Best Results

1. **Use full sentences** — The model works best with complete headlines or paragraphs.
2. **English only** — The model is trained on English-language news.
3. **More text = better accuracy** — A full paragraph gives more reliable results than 2–3 words.
4. **It analyzes writing patterns** — The model detects linguistic patterns, not factual accuracy. Use it as one tool among many.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "API Offline" in sidebar | The backend server is not running. Contact support. |
| No result appears | Make sure you entered text and clicked Analyze. |
| "Error" message after analysis | Try again with different text. If the problem persists, restart the system. |
| Page loads slowly | Wait a few seconds; the first load may take time. |
| Text box is too small | Drag the bottom-right corner of the text box to resize it. |

---

## Important Notes

- This tool provides an **AI-based prediction** and is not 100% accurate.
- Always verify important news from multiple trusted sources.
- The system does **not** store your submitted text.
- For technical questions or issues, contact the development team.

---

## Quick Reference

| Action | How |
|--------|-----|
| Analyze a news article | Paste text → Click Analyze |
| Try a sample | Click "Try Real News" or "Try Fake News" |
| Check system status | Look at sidebar → System Status |
| View this manual | Click "📖 User Manual" in sidebar |
