from embedding_generator_multilingual import generate_embeddings
import database_connection
import json

RETRIEVAL_LIMIT = 10

def retrieve_related_content(db_conn, query, should_print):
    cursor = db_conn.cursor()
    try:
        embeddings = generate_embeddings([(query, "dummy_source")], True)
        query_embedding = json.dumps(embeddings[0])

        cursor.execute(
            f"""SELECT id, content, source, 1 - (embedding <=> %s) AS cosine_similarity
               FROM items
               ORDER BY cosine_similarity DESC LIMIT {RETRIEVAL_LIMIT}""",
            (query_embedding,)
        )

        similar_content = []
        for row in cursor.fetchall():
            similar_content.append((row[1], row[2], row[3]))
            if (should_print):
                print((row[1], row[2], row[3]))

        return similar_content

    except Exception as e:
        print("Error executing query", str(e))
    finally:
        cursor.close()

# This check ensures that the function is only run when the script is executed directly, not when it's imported as a module.
# It can be used for running some quick tests and validations
if __name__ == "__main__":
    with database_connection.create_database_connection() as db_conn:
        retrieve_related_content(db_conn, "at which day did I get my first computer?", True)
