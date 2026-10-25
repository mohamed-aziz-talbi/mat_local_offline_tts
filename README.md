# 🖥️ Local Offline AI & Multi-TTS Web Assistant

A lightweight, 100% offline, privacy-first web application that brings local Large Language Models (LLMs) and multiple Text-to-Speech (TTS) engines together into a single, unified web dashboard. 

---

## 🌟 Key Features

* **🔒 100% Offline & Private:** Runs entirely on your local machine with zero data sent to external cloud APIs.
* **🌐 Unified Web Dashboard:** Clean, responsive single-page interface to interact with LLMs and test/generate audio across multiple TTS engines side-by-side.
* **🤖 Local LLM Integration:** Seamlessly connects to local model runners (such as Ollama or custom local endpoints) for real-time text generation and conversational assistance.
* **🔊 Multi-TTS Architecture:** Switch between and compare multiple local text-to-speech backends (e.g., F5-TTS, Sherpa-ONNX, and others) from the same interface.
* **⚡ Optimized Performance:** Designed for local hardware acceleration (CPU/GPU) to ensure low latency and smooth execution.

---

## 🛠️ Tech Stack

* **Backend:** Python, FastAPI / Flask (or your preferred framework)
* **Frontend:** HTML5, CSS3, JavaScript (or modern web stack)
* **AI / ML Ecosystem:** Ollama, PyTorch, ONNX Runtime, local TTS pipelines

---

## 🚀 Getting Started

### Prerequisites
Make sure you have the following installed on your local machine:
* Python 3.10+
* [Ollama](https://ollama.com/) (running locally for LLM support)
* Git

---

## 🐛 Known Issues & Active Development

I am actively tuning and optimizing the local pipeline. Current known behaviors being addressed include:
* **Audio Intermittence:** Occasional latency spikes when switching between different TTS backends on lower-spec hardware configurations.
* **Model Synchronization:** Context window handoffs between local LLMs and multi-engine TTS streaming require further optimization for real-time responsiveness.

If you encounter unexpected errors or have feedback, please check the [Issues](../../issues) tab or open a new bug report.

### Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/mat_local_offline_ts.git](https://github.com/your-username/mat_local_offline_ts.git)
   cd mat_local_offline_ts
