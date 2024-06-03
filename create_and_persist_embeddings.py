import database_connection
from embedding_generator_multilingual import generate_embeddings
import whatsapp_to_embeddings
import obsidian_to_embeddings
import json

BATCH_SIZE = 100 # Decrease this as needed, in case this puts too mcuh memeory pressure

def divide_into_chunks(target, n_size_per_chunk): 
    # Looping till length of target 
    for i in range(0, len(target), n_size_per_chunk):  
        yield target[i:i + n_size_per_chunk] 

def persist_content_list(db_conn, cursor, output_content_list):
    for chunk_idx, output_content_list_chunk in enumerate(divide_into_chunks(output_content_list, BATCH_SIZE)):
        chunk_info_message = f'{chunk_idx} of {len(output_content_list)//BATCH_SIZE}'
        print(f'Starting processing of chunk {chunk_info_message}')
        embeddings = generate_embeddings(output_content_list_chunk, False)
        for idx, content_and_source in enumerate(output_content_list_chunk):
            # print(idx, content_and_source)
            # Assumes that order is maintained
            embedding = embeddings[idx]
            embedding_json = json.dumps(embedding)
            content_sanitized = content_and_source[0].replace("\x00", "\uFFFD")
            cursor.execute("INSERT INTO items (content, source, embedding) VALUES (%s, %s, %s)", (content_sanitized, content_and_source[1], embedding_json))
        db_conn.commit()
        print(f'Persisted chunk {chunk_info_message}')

def run():
    with database_connection.create_database_connection() as db_conn:
        cursor = db_conn.cursor()
        try:        
            print(f'Starting persistence process for Obsidian...')
            persist_content_list(db_conn, cursor, obsidian_to_embeddings.parse(False))
            print(f'Starting persistence process for WhatsApp...')
            persist_content_list(db_conn, cursor, whatsapp_to_embeddings.parse(False))

        except Exception as e:
            print("Error executing query", str(e))
        finally:
            cursor.close()

if __name__ == "__main__":
    run()
