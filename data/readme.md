# UCI 30 dataset 

Version 0.1, June 14st 2019
Author: Chun Yang

## Summary
This package contains 30 datasets from UCI learning repository.
A. Frank and A. Asuncion, UCI machine learning repository, University of California, Irvine, School of Information and Computer Sciences, 2010. [Online]. Available: http://archive.ics.uci.edu/ml

## Details

| dataset      | Instances  | Attrib.    | Classes    | dataset      | Instances  | Attrib.    | Classes    |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|allbp         |3772        |29          |3           |artificial    |5109        |7           |10          |
|audiology     |226         |69          |24          |auto-mpg      |399         |7           |4           |
|autos         |205         |25          |7           |breast-w      |699         |9           |2           |
|clean1        |476         |166         |2           |colic         |368         |22          |2           |
|german        |1000        |24          |2           |glass         |214|9|7|
|heart-c       |303|13|5|heart-h       |294|13|5|
|imageseg      |2310|19|7|iris          |153    |4|3|
|labor         |57|6|2|led24         |3200|24|10|
|led7          |3200|7|10|lymph         |148|18|4|
|machine       |209|7|8|page-blocks   |5473|10|5|
|sickeuthyroid |3263|24|2|sonar         |208|60|2|
|splice        |3190 |61|3|tic-tac-toe   |958|9|2|
|vehicle       |848 |18|4|vote          |435|16|2|
|vowel         |990|13|11|wave21        |5000|21|3|
|wave40        |5000|40|3|zoo           |104|17|7|

All data sets are stored as Mat files.
In each Mat file, there are two parameters:

> input  : an NxNi matrix, where N is the number of data samples, Ni is the number of features;

> target : an Nx1 vector.
