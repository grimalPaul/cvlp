# Write config for a search

How the config works:

```json
{
    "kb_kwargs": {
        "path/to/kb  where you search": {
            "device": "indicate cpu or gpu",
            "index_kwargs": {
                "name_of_your_index_1": {
                    "key_kb": "column to index to do the search",
                    "index_load": "true or false if you want to load index from index path",
                    "index_path": "path where you want to load or save your index",
                    "string_factory": "https://github.com/facebookresearch/faiss/wiki/The-index-factory"
                },
                "name_of_your_index_2": {
                    "key_kb": "column to index to do the search",
                    "index_load": "true or false if you want to load index from index path",
                    "index_path": "path where you want to load or save your index",
                    "string_factory": "https://github.com/facebookresearch/faiss/wiki/The-index-factory"
                }
            }
        }
    }
}
```
