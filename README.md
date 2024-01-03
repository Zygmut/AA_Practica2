# Practica 2

L'objectiu d'aquesta segona pràctica es demostrar que heu assolit els conceptes que s'han explicar a l'assignatura i s'han practicat a les sessions presencials, relacionats amb el disseny i l'ús de xarxes neurals

## Condicions

1. El model que solucioni el problema estarà basat en xarxes neurals, aquestes s'han d'entrenar i avaluar emprant la llibreria Pytorch.

2. Es demana que com a mínim s'avaluin 2 models diferents: un que ha d'estar creat per vosaltres i un altre que es basi en una xarxa ja existent. Evidentment es permeten modificacions de la ja existent per adaptar-ho al problema que es vol resoldre.

3. El resultat del treball serà un informe on s'expliqui el procés que s'ha dut a terme per arribar a la que considereu que és millor solució. El document serà en format pdf. Podreu adjuntar una carpeta amb el codi i recursos que trobeu necessaris per comprovar la veracitat del que explicau al document.

4. Aquest document ha de tenir un llenguatge formal i tècnic i ha d'estar correctament estructurat:

    - Introducció al problema que es soluciona.
    - Solucions considerades (dades, característiques, models, mètriques).
    - Experiments realitzats.
    - Resultats dels experiments.
    - Conclusions.

5. A més del document explicatiu s'ha d'adjuntar (o enllaçar) un fitxer amb els pesos del millor entrenament de cada una de les xarxes que heu emprat (la que heu dissenyat vosaltres i la que ja existia), de tal manera que el professor pugui validar els resultats sense haver de repetir l'entrenament. Sense l'adjunció (enllaç) d'aquests fitxers la pràctica no es podrà aprovar.

## Avaluació

1. El treball es durà a terme en parelles.
2. El professor es reserva la possibilitat de convocar als grups a una revisió de la pràctica de forma presencial.
3. Per la xarxa que vosaltres dissenyeu, només està permés emprar tècniques de disseny i entrenament vistes a classe.
4. Tot el que no està fet pels alumnes ha d'estar referenciat, en cas contrari es considerarà com una còpia.
5. Per arribar a la solució és obligatori seguir les bones pràctiques de l'àrea a la preparació de les dades, entrenament i la validació dels resultats.

### Data d'entrega

Aquest treball es pot entregar fins el dia de l'examen a les 23:55 h tant per la convocatòria ordinaria com per l'extraordinària. Es realitzarà una tutoria dia XXXX de gener a les XXXX.

## Enunciat

Us proporciono el següent conjunt de dades: [The Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/). En paraules dels seus creadors: "Hem creat un conjunt de dades de mascotes de 37 categories amb unes 200 imatges per a cada classe. Les imatges tenen grans variacions d'escala, pose i il·luminació. Totes les imatges tenen una anotació associada a la raça, la posició del cap i la segmentació del mapa de nivell de píxels."

Amb aquest conjunt podeu realitzar 4 tasques diferents:

1. Classificació cans vs moixos: ha d'haver mínim una xarxa pròpia i un model ja existent. (màxim un 7)
2. Classificació de la raça (2 punt extra)
3. Detecció de la posició del cap (2 punt extra)
4. Segmentació de l'animal (punt extra)

## Installation

To complete this project, you will need to use [Conda](https://docs.conda.io/projects/miniconda/en/latest/) for managing the necessary Python libraries and dependencies.

### Conda environment

#### Creation

Use the following command to create a Conda environment for this project:

```bash
conda env create --prefix ./.env -f ./environment.yml
```

#### Export

To save the current Conda environment configuration for sharing or future use, execute the following command:

```bash
conda env export > conda_env.yml
```
