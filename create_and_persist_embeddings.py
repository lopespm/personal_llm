import database_connection
from embedding_generator_multilingual import generate_embeddings
import whatsapp_to_embeddings
import obsidian_to_embeddings
import json
import argparse

BATCH_SIZE = 512  # Decrease this if you hit memory pressure; larger batches are more GPU-efficient

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
    for chunk_idx, output_content_list_chunk in enumerate(chunks):
        chunk_info_message = f'{chunk_idx + 1} of {len(chunks)}'
        print(f'Starting processing of chunk {chunk_info_message}')
        embeddings = generate_embeddings(output_content_list_chunk, False)
        rows = [
            (content_and_source[0].replace("\x00", "\uFFFD"), content_and_source[1], json.dumps(embeddings[idx]))
            for idx, content_and_source in enumerate(output_content_list_chunk)
        ]
        cursor.executemany("INSERT INTO items (content, source, embedding) VALUES (%s, %s, %s)", rows)
        db_conn.commit()
        print(f'Persisted chunk {chunk_info_message}')

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
