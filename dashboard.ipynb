{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "eee0dc31-585e-4795-a44d-f7d474643e01",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting reportlab\n",
      "  Downloading reportlab-4.4.3-py3-none-any.whl.metadata (1.7 kB)\n",
      "Requirement already satisfied: pillow>=9.0.0 in ./Dev/Thesis/10 2/mp-env/lib/python3.10/site-packages (from reportlab) (11.3.0)\n",
      "Requirement already satisfied: charset-normalizer in ./Dev/Thesis/10 2/mp-env/lib/python3.10/site-packages (from reportlab) (3.4.2)\n",
      "Downloading reportlab-4.4.3-py3-none-any.whl (2.0 MB)\n",
      "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m2.0/2.0 MB\u001b[0m \u001b[31m5.2 MB/s\u001b[0m  \u001b[33m0:00:00\u001b[0m eta \u001b[36m0:00:01\u001b[0m\n",
      "Installing collected packages: reportlab\n",
      "Successfully installed reportlab-4.4.3\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install reportlab"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "a5ffc475-ae82-4f8d-a94c-31cd1e13b245",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from collections import Counter\n",
    "import glob\n",
    "import os\n",
    "from reportlab.lib.pagesizes import letter\n",
    "from reportlab.pdfgen import canvas"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "2be09b01-f4bb-4bcd-acd3-7edd70fe744a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Constants\n",
    "EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']\n",
    "TRAIN_PATH = \"/Users/nazmussakibsheam/GUB/Data-mining-lab/face_expression_detector/webcam_dataset/train\"\n",
    "EMOTION_HISTORY_PATH = 'emotion_history.csv'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "f1ec1642-110e-4c1c-90b5-f49768e63dff",
   "metadata": {},
   "outputs": [],
   "source": [
    "def generate_pdf_report(df):\n",
    "    \"\"\"Generate a PDF report with emotion statistics and plots.\"\"\"\n",
    "    pdf_path = 'emotion_report.pdf'\n",
    "    c = canvas.Canvas(pdf_path, pagesize=letter)\n",
    "    c.setFont(\"Helvetica\", 12)\n",
    "    c.drawString(100, 750, \"Facial Expression Recognition Report\")\n",
    "    \n",
    "    # Emotion distribution\n",
    "    c.drawString(100, 700, \"Emotion Distribution:\")\n",
    "    emotion_counts = Counter(df['Emotion'])\n",
    "    y_pos = 680\n",
    "    for emo, count in emotion_counts.items():\n",
    "        c.drawString(120, y_pos, f\"{emo}: {count}\")\n",
    "        y_pos -= 20\n",
    "\n",
    "    # Include plots\n",
    "    if os.path.exists('confusion_matrix.png'):\n",
    "        c.drawImage('confusion_matrix.png', 100, 400, width=400, height=300)\n",
    "    if os.path.exists('training_history.png'):\n",
    "        c.drawImage('training_history.png', 100, 100, width=400, height=200)\n",
    "    \n",
    "    c.save()\n",
    "    return pdf_path"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "46bce1c0-68cb-46e2-9120-d9405eed9ff6",
   "metadata": {},
   "outputs": [],
   "source": [
    "def streamlit_dashboard():\n",
    "    \"\"\"Launch Streamlit dashboard to visualize emotion trends.\"\"\"\n",
    "    st.title(\"Facial Expression Recognition Dashboard\")\n",
    "    st.write(\"Analyze emotions detected during the webcam session.\")\n",
    "\n",
    "    if not os.path.exists(EMOTION_HISTORY_PATH):\n",
    "        st.error(\"No emotion history found. Run face_expression_detector.ipynb first.\")\n",
    "        return\n",
    "\n",
    "    df = pd.read_csv(EMOTION_HISTORY_PATH)\n",
    "    if df.empty:\n",
    "        st.warning(\"No emotion data available.\")\n",
    "        return\n",
    "\n",
    "    # Sort by timestamp to ensure monotonic index\n",
    "    df['Timestamp'] = pd.to_datetime(df['Timestamp'])\n",
    "    df = df.sort_values('Timestamp')\n",
    "\n",
    "    # Emotion Timeline\n",
    "    st.subheader(\"Emotion Timeline\")\n",
    "    df['TimeBin'] = df['Timestamp'].dt.floor('S')  # Bin by second\n",
    "    timeline_data = df.groupby('TimeBin')['Emotion'].value_counts().unstack().fillna(0)\n",
    "    st.line_chart(timeline_data)\n",
    "\n",
    "    # Emotion Distribution\n",
    "    st.subheader(\"Emotion Distribution\")\n",
    "    emotion_counts = Counter(df['Emotion'])\n",
    "    st.bar_chart(emotion_counts)\n",
    "\n",
    "    # Captured Images\n",
    "    st.subheader(\"Captured Images\")\n",
    "    cols = st.columns(3)\n",
    "    for i, img_path in enumerate(glob.glob(os.path.join(TRAIN_PATH, '*/*.jpg'))[:9]):\n",
    "        with cols[i % 3]:\n",
    "            st.image(img_path, caption=os.path.basename(os.path.dirname(img_path)))\n",
    "\n",
    "    # Download CSV\n",
    "    st.subheader(\"Download CSV Report\")\n",
    "    st.download_button(\"Download CSV\", df.to_csv(index=False), \"emotion_report.csv\", \"text/csv\")\n",
    "\n",
    "    # Download PDF\n",
    "    st.subheader(\"Download PDF Report\")\n",
    "    pdf_path = generate_pdf_report(df)\n",
    "    with open(pdf_path, \"rb\") as f:\n",
    "        st.download_button(\"Download PDF\", f, \"emotion_report.pdf\", \"application/pdf\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "3df269bf-05b9-4bfe-9e06-1dc5631fe3ba",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-08-09 20:35:05.871 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-09 20:35:05.906 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run /Users/nazmussakibsheam/Dev/Thesis/10 2/mp-env/lib/python3.10/site-packages/ipykernel_launcher.py [ARGUMENTS]\n",
      "2025-08-09 20:35:05.906 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-09 20:35:05.907 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-09 20:35:05.907 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-09 20:35:05.907 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-09 20:35:05.908 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-09 20:35:05.908 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-09 20:35:05.908 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-09 20:35:05.908 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "if __name__ == \"__main__\":\n",
    "    streamlit_dashboard()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "862af816-ed28-4994-9655-5878b119f887",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3.10 (mp-env)",
   "language": "python",
   "name": "mp-env"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.18"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
