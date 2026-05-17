import database_connection
from embedding_generator_multilingual import generate_embeddings
import whatsapp_to_embeddings
import obsidian_to_embeddings
import json
import argparse
import threading
import queue
from tqdm import tqdm

BATCH_SIZE = 64  # Fine on 128 GB unified memory with max_length=1024 (attention per layer ~2 GB)

def divide_into_chunks(target, n_size_per_chunk): 
    # Looping till length of target 
    for i in range(0, len(target), n_size_per_chunk):  
        yield target[i:i + n_size_per_chunk] 

def get_already_processed_sources(cursor):
    cursor.execute("SELECT DISTINCT source FROM items")
    return {row[0] for row in cursor.fetchall()}

def persist_content_list(db_conn, cursor, output_content_list, already_processed_sources):
    pending = [(content, source) for content, source in output_content_list if source not in already_processed_sources]
    skipped = len(output_content_list) - len(pending)
    if skipped > 0:
        print(f'Skipping {skipped} already-processed items')
    chunks = list(divide_into_chunks(pending, BATCH_SIZE))
    total_chunks = len(chunks)

    # Pipeline: DB writes run in a background thread so the GPU never waits on network I/O.
    _DONE = object()
    write_queue = queue.Queue(maxsize=2)  # back-pressure: don't let embeddings race too far ahead
    write_errors = []

    def db_writer():
        while True:
            item = write_queue.get()
            if item is _DONE:
                break
            rows, chunk_info_message = item
            try:
                cursor.executemany("INSERT INTO items (content, source, embedding) VALUES (%s, %s, %s)", rows)
                db_conn.commit()
                tqdm.write(f'  ✓ Persisted chunk {chunk_info_message}')
            except Exception as e:
                write_errors.append(e)
                break

    writer = threading.Thread(target=db_writer, daemon=True)
    writer.start()
    try:
        for chunk_idx, chunk in enumerate(tqdm(chunks, desc='Embedding batches', unit='batch')):
            chunk_info_message = f'{chunk_idx + 1} of {total_chunks}'
            embeddings = generate_embeddings(chunk, False)
            rows = [
                (item[0].replace("\x00", "\uFFFD"), item[1], json.dumps(embeddings[idx]))
                for idx, item in enumerate(chunk)
            ]
            write_queue.put((rows, chunk_info_message))
            if write_errors:
                raise write_errors[0]
    finally:
        write_queue.put(_DONE)
        writer.join()
    if write_errors:
        raise write_errors[0]

def run(wipe=False):
    with database_connection.create_database_connection() as db_conn:
        cursor = db_conn.cursor()
        try:
            if wipe:
                print('Wiping items table...')
                cursor.execute("DELETE FROM items")
                db_conn.commit()
                print('Items table wiped.')
            already_processed_sources = get_already_processed_sources(cursor)
            print(f'Starting persistence process for Obsidian...')
            persist_content_list(db_conn, cursor, obsidian_to_embeddings.parse(False), already_processed_sources)
            print(f'Starting persistence process for WhatsApp...')
            persist_content_list(db_conn, cursor, whatsapp_to_embeddings.parse(False), already_processed_sources)

        except Exception as e:
            print("Error executing query", str(e))
        finally:
            cursor.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wipe', action='store_true', help='Wipe the items table before processing')
    args = parser.parse_args()
    run(wipe=args.wipe)
