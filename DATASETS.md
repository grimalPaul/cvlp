# Datasets

"triviaqa"
"match_image"
"match_article"
"viquae"

## multimedia and wikimage

961 entities in test
82 entities are present in multimedia and wikimage

2397 unique entity in viquae
217 are in multimedia and wikimage

Should I delete this data  ?

I dont think so, bc different tasks

voir combien de question sont impliqués

for multimedia we filter to be sure to have 2 differents passages
d'enlever 7164
il reste 36578

On enlève les entités en communs avec viquae

Pas d'entités en communs avec le dataset Viquae

- Pour Multimedia : on a 36361 entités avec des passages
- Pour Wikimage: on a 43525 entités avec au moins deux images

Puis je shuffle et split wikimage
Je recupéère les clés du split
et je filtre multimedia pour avoir les mêmes entités dans validation et dans train que dans wikimage(pas le même nombre car on a enlevé les articles ou on avait qu'un unique passage)

```java
{
    '2': 36938,
    '3': 4205, 
    '4': 1356, 
    '6': 198, 
    '9': 40, 
    '5': 478, 
    '7': 110, 
    '12': 15, 
    '8': 76, 
    '11': 23, 
    '19': 3, 
    '10': 31, 
    '17': 7, 
    '22': 4, 
    '16': 10, 
    '15': 7, 
    '14': 6, 
    '23': 2, 
    '27': 1, 
    '41': 2, 
    '18': 4, 
    '24': 2, 
    '13': 5, 
    '26': 1, 
    '40': 1
    }
```




après split on a
multimedia :
train :  num_rows: 33871
validation :  num_rows: 2490

wikimage:
train:40525
validation:3000



on cherche dan sl a kb, entités qui on au moins deux image 
on crée multimedia et wikimage
on supprime dans multimedia entités qui n'ont pas au moins deux passages
on filtre les deux datasets de toutes les entités présentent dans viquae
On shuffle puis on split wikimage sur un nombre d'entités, on utilise les mêmes entités pour faire le split de multimedia