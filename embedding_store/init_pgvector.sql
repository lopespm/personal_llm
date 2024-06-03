-- Install the extension we just compiled
CREATE EXTENSION IF NOT EXISTS vector;

/*
For simplicity, we are directly adding the content into this table as
a column containing text data. It could easily be a foreign key pointing to
another table instead that has the content you want to vectorize for
semantic search, just storing here the vectorized content in our "items" table.

"1024" dimensions for our vector embedding is critical - that is the
number of dimensions our embeddings model output
*/

CREATE TABLE IF NOT EXISTS items (id bigserial PRIMARY KEY, content TEXT, source TEXT, embedding vector(1024));