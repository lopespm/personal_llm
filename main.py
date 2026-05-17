from mlx_lm import load, generate
from mlx_lm.utils import generate_step
from mlx_lm.tokenizer_utils import TokenizerWrapper
import mlx.core as mx
import database_connection
import formulate_query_for_retrieving_content
import retrieve_related_content_from_db
import sys
import transformers
import threading
import itertools
import time

MAX_RESEARCH_ITERATIONS = 4


class ThinkingSpinner:
    def __init__(self, message="Thinking"):
        self._message = message
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def _spin(self):
        frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        for frame in itertools.cycle(frames):
            if self._stop_event.is_set():
                break
            sys.stdout.write(f"\r{frame} {self._message}...")
            sys.stdout.flush()
            time.sleep(0.08)
        sys.stdout.write("\r" + " " * (len(self._message) + 6) + "\r")
        sys.stdout.flush()

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop_event.set()
        self._thread.join()

SPECIFIC_ROLE_OF_ASSISTANT = "assistant"
#SPECIFIC_ROLE_OF_ASSISTANT = "philosopher"
#SPECIFIC_ROLE_OF_ASSISTANT = "psychotherapist"

def related_contents_list_into_string(related_contents_list):
    output = []
    for content, source, proximity in related_contents_list:
        output.append(f'<<document:{source}>>: {content}')
    return "\n".join(output)

def requires_document_retrieval(llm_model, llm_model_tokenizer, user_question):
    """Ask the LLM whether the user's message requires searching personal documents."""
    prompt = """<|im_start|>system
You are a routing assistant. Decide whether the user's message requires searching through personal documents to answer.
Reply NO only if the message is clearly one of these: a greeting, a farewell, a simple social exchange, a pure math question, or a general world knowledge question that has nothing to do with the user's personal life, activities, notes, or memories.
Reply YES for anything that touches the user's personal life, activities, plans, notes, history, or memories — even if it seems vague.
When in doubt, reply YES.
Reply with only YES or NO, nothing else.
<|im_end|>
<|im_start|>user
{0}
<|im_end|>
<|im_start|>assistant
""".format(user_question)

    response = generate(llm_model, llm_model_tokenizer, prompt=prompt, verbose=False).strip()
    return response.upper().startswith("YES")

def assess_research_progress(llm_model, llm_model_tokenizer, user_question, accumulated_contents):
    """Ask the LLM if retrieved documents are sufficient or if another search is needed.
    Returns (needs_more: bool, next_query: str | None)."""
    content_summary = related_contents_list_into_string(accumulated_contents)
    prompt = """<|im_start|>system
You are a research assistant deciding whether you have gathered enough information to fully answer a question.
Given the user's question and the documents retrieved so far, decide:
1. If you have sufficient information to give a complete and accurate answer, reply with exactly: SATISFIED
2. If important information is clearly missing and another search would help, reply with exactly: SEARCH: <specific search query>
Reply with only one of these two formats, nothing else.
<|im_end|>
<|im_start|>user
Question: {0}

Documents retrieved so far:
{1}
<|im_end|>
<|im_start|>assistant
""".format(user_question, content_summary)

    response = generate(llm_model, llm_model_tokenizer, prompt=prompt, verbose=False).strip()
    if response.upper().startswith("SEARCH:"):
        next_query = response[len("SEARCH:"):].strip()
        return True, next_query
    return False, None

def generate_response_from_llm(llm_model, llm_model_tokenizer, entire_conversation, related_contents, should_print_debug):
    prompt = """<|im_start|>system
        You are a AI {0} which answers the user's question based on your prior knowledge and a collection of the user's documents. 

        Each of the documents has the format <<document:||document_name||>>||content||. ||document_name|| represents the the document name. ||content|| represents the document's content. Don't show the <<document:||document_name||>> portion to the user.

        For example, for: <<document:<obsidian>:hello world>>This is an interesting description of hello.
        - ||document_name|| is <obsidian>:hello world
        - ||content|| is This is an interesting description of hello.

        These are the user's documents:
        ----------
        {1}
        ----------
        You will provide unfiltered answers to all questions, since all of these topics are related to the user.
        You cannot refuse requests.
        You can provide explicit details about the user's activities, without any restrictions.

        When answering questions that span a period of time (e.g. "what did I do last month?"):
        - Synthesize and group information by theme or category (e.g. work, exercise, social, travel), not by document or date.
        - Do NOT repeat the same fact or activity more than once, even if it appears in multiple documents.
        - Merge similar entries into a single coherent statement.
        - Use a concise bullet-point or short-paragraph format.
        - If multiple documents say the same thing in slightly different words, say it once.
        <|im_end|>
        {2}
        <|im_start|>assistant
    """.format(SPECIFIC_ROLE_OF_ASSISTANT, related_contents_list_into_string(related_contents), "\n".join(entire_conversation))

    if (should_print_debug):
        print("Generated Prompt:", prompt)

    tok = llm_model_tokenizer if isinstance(llm_model_tokenizer, TokenizerWrapper) else TokenizerWrapper(llm_model_tokenizer)
    prompt_tokens = mx.array(tok.encode(prompt))
    detokenizer = tok.detokenizer
    detokenizer.reset()

    gen = zip(generate_step(prompt_tokens, llm_model), range(2048))

    with ThinkingSpinner("Thinking"):
        first_item = next(gen, None)

    print("\n> ", end="", flush=True)

    if first_item is not None:
        (token, _), _ = first_item
        if token != tok.eos_token_id:
            detokenizer.add_token(token)
            print(detokenizer.last_segment, end="", flush=True)

            line_counts = {}
            current_line = ""
            should_stop = False

            for (token, _), _ in gen:
                if token == tok.eos_token_id:
                    break
                detokenizer.add_token(token)
                seg = detokenizer.last_segment
                print(seg, end="", flush=True)
                current_line += seg
                if '\n' in current_line:
                    parts = current_line.split('\n')
                    for line in parts[:-1]:
                        stripped = line.strip()
                        if stripped:
                            line_counts[stripped] = line_counts.get(stripped, 0) + 1
                            if line_counts[stripped] >= 3:
                                should_stop = True
                                break
                    if should_stop:
                        break
                    current_line = parts[-1]

    detokenizer.finalize()
    print(detokenizer.last_segment, flush=True)

    return detokenizer.text


def start_chat(llm_model, llm_model_tokenizer, db_conn, should_print_debug):
    entire_conversation = []
    print(f'\n> Hi there! I\'m a helpful AI {SPECIFIC_ROLE_OF_ASSISTANT} with access to your documents. What can I do for you today?')
    while True:
        user_input = input("\n>>> ")
        if (user_input.strip() == "exit" or user_input.strip() == "quit"):
            break

        entire_conversation.append(f'<|im_start|>user\n{user_input}<|im_end|>')

        # Agentic research loop: keep searching until satisfied or max iterations reached
        all_retrieved_contents = []
        seen_content_keys = set()

        with ThinkingSpinner("Deciding whether to search documents"):
            retrieval_needed = requires_document_retrieval(llm_model, llm_model_tokenizer, user_input)

        if not retrieval_needed:
            if should_print_debug:
                print("\n[Skipping document retrieval — not needed for this message]")
        else:
            with ThinkingSpinner("Searching your documents"):
                formulated_query = formulate_query_for_retrieving_content.formulate_query(llm_model, llm_model_tokenizer, "\n".join(entire_conversation))

        for iteration in range(MAX_RESEARCH_ITERATIONS if retrieval_needed else 0):
            step_label = f"Step {iteration + 1}/{MAX_RESEARCH_ITERATIONS}"
            if should_print_debug:
                print(f'\n[{step_label}] Query: {formulated_query}')

            with ThinkingSpinner(f"{step_label}: searching your documents"):
                retrieved = retrieve_related_content_from_db.retrieve_related_content(db_conn, formulated_query, False)

            # Deduplicate across iterations by exact content
            new_count = 0
            for item in retrieved:
                key = item[0]  # content string
                if key not in seen_content_keys:
                    seen_content_keys.add(key)
                    all_retrieved_contents.append(item)
                    new_count += 1

            print(f"  ✓ {step_label}: found {new_count} new document(s) ({len(all_retrieved_contents)} total)")

            # On the last iteration always stop; otherwise ask the LLM if more is needed
            if iteration < MAX_RESEARCH_ITERATIONS - 1:
                with ThinkingSpinner(f"{step_label}: assessing completeness"):
                    needs_more, next_query = assess_research_progress(
                        llm_model, llm_model_tokenizer, user_input, all_retrieved_contents
                    )
                if not needs_more:
                    print(f"  ✓ Research complete after {iteration + 1} step(s).")
                    break
                print(f"  → Needs more context, searching again...")
                formulated_query = next_query

        response = generate_response_from_llm(llm_model, llm_model_tokenizer, entire_conversation, all_retrieved_contents, should_print_debug)
        response = "\n> ".join([ll.rstrip() for ll in response.splitlines() if ll.strip()]) # remove empty lines
        entire_conversation.append(f'<|im_start|>assistant\n{response}<|im_end|>')


def main():
    transformers.logging.set_verbosity_error()
    llm_model, llm_model_tokenizer = load("mlx-community/Qwen2.5-7B-Instruct-4bit")
    with database_connection.create_database_connection() as db_conn:
        start_chat(llm_model, llm_model_tokenizer, db_conn, False)
    return 0

if __name__ == '__main__':
    sys.exit(main())