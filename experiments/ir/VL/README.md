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
                    "index_path": "where you want to save the index or load index. Be carefull exact name of the index when you load it but when you save it the name will be path/name_of_your_index_1.faiss",
                    "string_factory": "https://github.com/facebookresearch/faiss/wiki/The-index-factory"
                },
                "name_of_your_index_2": {
                    "key_kb": "column to index to do the search",
                    "index_load": "true or false if you want to load index from index path",
                    "index_path": "Be carefull exact name of the index when you load it but when you save it the name will be path/name_of_your_index_1.faiss",
                    "string_factory": "https://github.com/facebookresearch/faiss/wiki/The-index-factory"
                }
            }
        }
    }
}
```
