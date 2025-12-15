## 📝 Generative AI Use Case: Summarize Dialogue Readme

This notebook, `Lab_1_summarize_dialogue.ipynb`, provides a practical lab for exploring **dialogue summarization** using Generative AI, specifically the **FLAN-T5 Large Language Model (LLM)**.

The core objective is to understand how the prompt supplied to the model (a concept known as **Prompt Engineering**) significantly influences the quality of the generated summary. The lab compares three primary inference techniques: **Zero Shot**, **One Shot**, and **Few Shot** learning.

---

### 📚 Table of Contents

1.  **Set up Kernel and Required Dependencies**
    * Verification of the required compute instance type (`ml.m5.2xlarge`).
    * Installation and setup of necessary Python packages, including `datasets`, `transformers`, `torch`, `peft`, etc.
    * Loading core components: `datasets`, `AutoModelForSeq2SeqLM`, `AutoTokenizer`, and `GenerationConfig`.
2.  **Summarize Dialogue without Prompt Engineering**
    * Loading sample dialogues and their baseline human summaries from the **DialogSum** Hugging Face dataset.
    * Loading the pre-trained **`google/flan-t5-base`** model and its tokenizer.
    * Demonstrating the need for prompt engineering by observing the model's output when the raw dialogue is used as the only input. The model initially struggles to identify the summarization task.
3.  **Summarize Dialogue with an Instruction Prompt**
    * **Zero Shot Inference with an Instruction Prompt**: Wrapping the dialogue in an explicit instruction (e.g., "Summarize the following conversation.") to direct the model's behavior. This shows a qualitative improvement over the base model.
    * **Zero Shot Inference with the Prompt Template from FLAN-T5**: Using a known, effective prompt structure for T5-based models (e.g., `Dialogue: ... What was going on?`) to further guide the model.
4.  **Summarize Dialogue with One Shot and Few Shot Inference**
    * **One Shot Inference**: Providing the model with **one** complete example of an input dialogue and its expected summary *before* the dialogue to be summarized. This "in-context learning" significantly improves the summary quality compared to zero-shot methods.
    * **Few Shot Inference**: Providing the model with **multiple** complete dialogue-summary examples before the target dialogue. This is shown to potentially offer marginal or no further improvement over one-shot, and users must be mindful of the model's input-context length (e.g., 512 tokens for FLAN-T5).
5.  **Generative Configuration Parameters for Inference**
    * Exploring the use of the `GenerationConfig` class to adjust parameters that influence the LLM's output.
    * Experimenting with parameters like `max_new_tokens` and decoding strategies such as **sampling** (`do_sample = True`) with controlled randomness using **`temperature`**.

---

### ✨ Key Concepts Explored

| Concept | Description |
| :--- | :--- |
| **Prompt Engineering** | The practice of designing the input text (**prompt**) to effectively guide the LLM toward generating the desired output for a specific task. |
| **Tokenization** | The process of splitting raw text into smaller units (tokens) that the LLM can understand and process. |
| **Zero Shot Inference** | Performing a task by giving the LLM an instruction, **without** providing any examples of the task (e.g., "Summarize this..."). |
| **One Shot Inference** | Performing a task by giving the LLM a single full example of the task's input and desired output *before* the instruction for the new input. |
| **Few Shot Inference** | Performing a task by giving the LLM multiple full examples of the task's input and desired output *before* the new input. |
| **In-Context Learning** | The phenomenon where an LLM learns a new task simply by being shown examples directly in the input prompt (used in one-shot and few-shot inference). |
| **GenerationConfig** | A class from the Hugging Face `transformers` library used to manage parameters like `max_new_tokens` and `temperature` during the text generation process. |

---

### 🛠️ How to Run the Notebook

1.  **Environment Setup:** Ensure you are running on a machine with sufficient resources, as the lab verifies the use of an instance like `ml.m5.2xlarge`.
2.  **Dependencies:** Run the setup cells to install or upgrade necessary packages, including `tensorflow`, `keras`, `torch`, `datasets`, and `transformers`.
3.  **Execution:** Run the cells sequentially to load the dataset, model, and tokenizer, and then observe the comparative results of the various prompt engineering techniques.
