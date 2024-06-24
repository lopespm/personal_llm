# Personal Local LLM (Llama3): powered by WhatsApp and Obsidian Data

![Personal Local LLM (Llama3): powered by WhatsApp and Obsidian Data Intro Image](https://github.com/lopespm/agent-trainer/assets/3640622/87bc19b6-268d-4a1e-90f8-a16ce34c80c3)
<intro>

More details **[in this blog article](https://lopespm.github.io/machine_learning/2024/06/24/personal-llm.html)**

## Requirements

- Computer with Apple Silicon (this project uses MLX)
- Conda
- Docker and Docker Compose

## Usage

### 1. Set environment / root variables
Edit the .env file, which will define the data access to the embedding store (POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB):
```bash
$ nano embedding_store/.env
```

Edit the document base paths used :
```bash
# Change OBSIDIAN_VAULT_PATH to the root folder of the vaults you wish to add to the database
$ nano obsidian_to_embeddings.py
# Change WHATSAPP_WORKING_PATH to the folder where you have exported your WhatsApp messages data (result.json) and your contacts (contacts.csv) [see below for more details]
$ nano whatsapp_to_embeddings.py
```

### 2. Start the embeddings DB and Python environment
Start the docker container which will host the postgres DB:
```bash
$ cd embedding_store
$ docker compose up
```
Set up the conda environment, for running our python scripts:

```bash
# Create the conda environment (only need to do this once)
$ conda env create -f environment.yml
# Activate it
$ conda activate personal_llm
```

### 3. Populate embeddings DB with the WhatsApp and Obsidian contents

Parse your WhatsApp and Obsidian information, and store it as embeddings in the DB (you only need to run this once, and this might take a while) 
```bash
$ python create_and_persist_embeddings.py
```

### 4. Start the chat with the LLM!

```bash
$ python main.py
```

This starts the chat, and you can expect to see something like this:
```
>>> Hi there! I'm a helpful AI assistant with access to your documents. What can I do for you today?

> |
```

## Setup of Data Sources

### WhatsApp
We will need to files, in order to get the most benefit of 
- result.json (dump of all your WhatsApp messages)
- contacts.csv (dump of all yor [Google] contacts, which will be used to enrich the messages above; you can opt out from using it, but you will get less benefit from messaging information)

#### result.json
Use [WhatsApp-Chat-Exporter](https://github.com/KnugiHK/WhatsApp-Chat-Exporter) to export all of the messaging information from your WhatsApp repo. Follow the steps on that repo for more details. These are overall:
1. Trigger a chat backup via the app
2. Copy the entire WhatsApp folder from your mobile device (on Android, it is located at `Android/media/com.whatsapp/WhatsApp`), into your computer
4. Extract `msgstore.db` as mentioned [here](https://github.com/KnugiHK/WhatsApp-Chat-Exporter?tab=readme-ov-file#unencrypted-whatsapp-database)
4. Run `pip install whatsapp-chat-exporter`
5. In the WhatsApp working folder where you have `msgstore.db`, `WhatsApp` folder and contact database `wa.db`, run `wtsexporter -a --json`
6. This will output a `result.json` file. Copy it to the WHATSAPP_WORKING_PATH folder mentioned above


#### contacts.csv
1. Go to https://contacts.google.com/
2. Export all your contacts. Select "CVS from Google" option
3. This will output a contacts.csv file. Copy it to the WHATSAPP_WORKING_PATH folder mentioned above

### Obsidian
Nothing required to be setup. Since Obsidian stores its text contents inside markdown (.md) files, we just need to point directly to the folder we want to target
