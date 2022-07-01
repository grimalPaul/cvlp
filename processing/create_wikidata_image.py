import json 
from datasets import load_from_disk, disable_caching
import sys
from SPARQLWrapper import SPARQLWrapper, JSON

disable_caching()

class Searcher(object):
    def __init__(self):
        user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
        endpoint_url = "https://query.wikidata.org/sparql"
        self.sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
        

    def request(self,query):
        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)
        return self.sparql.query().convert()

def create_query(keys):
    sparql_values = list(map(lambda id: "wd:" + id,keys))
    query = '''
    SELECT * WHERE{ 
        VALUES ?item { %s } 
        ?item wdt:P18 ?image. }
        ''' % " ".join(sparql_values)
    print(query)
    return query

def format_results(elements, results):
    for item in results["results"]["bindings"]:
        index = item['item']['value']
        url_image = item['image']['value']
        wikidata_id = index.split('/')[-1]
        if wikidata_id not in elements:
            elements[wikidata_id] = []
        elements[wikidata_id].append(url_image)
    return elements

def worker(keys, step_size = 10000):
    size = len(keys)
    nb_iteration =  size//step_size
    remaining_elements = size%step_size
    print(f'size {size}, iter {nb_iteration}, remaining elements {remaining_elements}')
    data = {}
    searcher = Searcher()
    for i in range(nb_iteration):
        query = create_query(keys[:step_size])
        keys = keys[step_size:]
        r = searcher.request(query)
        data = format_results(data, r)

    if remaining_elements > 0:
        query = create_query(keys)
        r = searcher.request(query)
        data = format_results(data, r)
    
    # check if at least 2 images
    dataset = {}
    cpt = 0
    for id,l in data.items():
        if len(l) > 1:
            r[id] = l
            cpt+=1
    print(cpt)

    # save dataset
    with open('data/wikimage.json', 'w') as fp:
        json.dump(dataset, fp)

if __name__ == '__main__':
    path_dataset = "data/wikimage"
    dataset = load_from_disk(path_dataset)
    keys = dataset["wikidata_id"]

    keys = [
        "Q2453276",
        "Q157986",
        "Q25173",
        "Q76",
        "Q39476",
        "Q312",
        "Q19837",
        "Q36301"
    ]
    worker(keys)