# Fine Tunning

https://platform.openai.com/docs/guides/fine-tuning

## Requirements:

1. Python installed on your machine
2. Valid Open AI API key
 
### STEP 1) Prepare your Dataset
https://platform.openai.com/docs/guides/fine-tuning/data-formatting

To fine-tune a model, you'll need a set of training examples that each consist of a single input ("prompt") and its associated output ("completion")

For this lab we will create a dataset using ChatGPT

- We used to following prompt

``` bash
Write a Python script that creates a CSV with the following columns.
Country, Population, Language, PIB, IPC, Weather, Flag Colors, Religion, Poverty %, President

Create a sample of 10 countries with the data on the headers
```

- The python script returned by ChatGPT 

``` python
import csv
import random

# Sample data for each column
countries = ["United States", "China", "Russia", "Brazil", "India", "Canada", "Mexico", "France", "Germany", "Japan", "Australia", "South Korea", "Egypt", "Nigeria", "South Africa","Spain", "Italy", "Netherlands", "Argentina", "Chile", "Thailand", "Indonesia", "Poland", "Sweden", "Switzerland", "Norway", "Finland", "Denmark", "Ireland", "Portugal", "Austria", "Belgium", "Greece", "Czech Republic", "Hungary"]
populations = [331000000, 1386000000, 144400000, 213000000, 1390000000,37742154, 130222814, 67413000, 83190556, 126476461, 25687041, 51709098, 104258327, 211400708, 60041925, 46754778, 60252824, 17173000, 45376763, 19458310, 69799978, 271350000, 38476957, 10175214, 8715625, 5367580, 5540720, 5818553, 4982900, 10276617, 8917205, 11589623, 10746740, 10708981, 9745900]
languages = ["English", "Mandarin", "Russian", "Portuguese", "Hindi, English", "English, French", "Spanish", "French", "German", "Japanese", "English", "Korean", "Arabic", "English", "Zulu, Xhosa, Afrikaans", "Spanish", "Italian", "Dutch", "Spanish", "Spanish", "Thai", "Indonesian", "Polish", "Swedish", "German, French, Italian, Romansh", "Norwegian", "Finnish", "Danish", "English, Irish", "Portuguese", "German", "Dutch, French, German", "Greek", "Czech", "Hungarian"]
PIB = [21433204, 14342903, 1699861, 2143872, 3054218, 1737177, 1212241, 2844985, 4135497, 5172526, 1532089, 1782355, 303626, 448125, 296219, 1368196, 2108546, 912924, 451338, 298796, 543605, 1130441, 614319, 538957, 705874, 403212, 236225, 306118, 393875, 230427, 482842, 528207, 209853, 248625, 173025]
IPC = [63.4, 8.3, 11.2, 10.1, 7.3, 10.9, 8.3, 8.9, 9.6, 9.9, 7.4, 10.7, 4.0, 1.9, 4.1, 8.6, 8.5, 10.4, 9.9, 7.1, 4.7, 4.1, 7.2, 10.2, 9.0, 11.0, 10.7, 10.6, 10.9, 9.5, 9.9, 10.2, 8.2, 9.5, 9.7]
weather = ["Temperate", "Temperate", "Temperate", "Tropical", "Tropical", "Temperate", "Tropical", "Temperate", "Temperate", "Temperate", "Temperate", "Temperate", "Desert", "Tropical", "Temperate", "Mediterranean", "Mediterranean", "Temperate", "Temperate", "Temperate", "Tropical", "Tropical", "Temperate", "Temperate", "Temperate", "Temperate", "Temperate", "Temperate", "Temperate", "Mediterranean", "Temperate", "Temperate", "Mediterranean", "Temperate", "Temperate"]
flag_colors = ["Red, White, Blue", "Red, Yellow", "White, Blue, Red", "Green, Yellow, Blue", "Saffron, White, Green", "Red, White", "Green, White, Red", "Blue, White, Red", "Black, Red, Gold", "White, Red", "Blue, White, Red", "White, Black, Red", "Black, White, Red", "Green, White, Green", "Black, White, Green", "Red, Yellow", "Green, White, Red", "Red, White, Blue", "Blue, White", "Red, White, Blue", "Red, White, Blue", "Red, White", "White, Red", "Blue, Yellow", "Red, White", "Red, White, Blue", "Blue, White", "Red, White", "Green, White, Orange", "Green, Red", "Red, White, Red", "Black, Yellow, Red", "Blue, White", "Red, White, Blue", "Red, White, Green"]
religion = ["Christianity", "Irreligion, Buddhism, Taoism, Confucianism", "Russian Orthodoxy", "Roman Catholicism", "Hinduism, Islam", "Christianity", "Catholicism", "Christianity", "Christianity", "Shintoism, Buddhism", "Christianity", "Christianity, Buddhism", "Islam", "Christianity", "Christianity", "Christianity", "Christianity", "Christianity", "Christianity", "Christianity", "Buddhism", "Islam", "Christianity", "Christianity", "Christianity", "Christianity", "Christianity", "Christianity", "Christianity", "Christianity", "Christianity", "Christianity", "Christianity", "Christianity", "Christianity"]
poverty_percent = [9.2, 3.1, 12.3, 21.2, 21.9, 9.5, 41.9, 8.5, 14.8, 16.1, 13.6, 13.5, 32.5, 40.1, 25.2, 21.5, 10.9, 8.6, 32.0, 8.6, 7.8, 9.2, 3.5, 8.3, 6.2, 4.6, 5.5, 6.2, 12.8, 17.3, 3.3, 6.0, 34.0, 9.6, 11.6]
president = ["Joe Biden", "Xi Jinping", "Vladimir Putin", "Jair Bolsonaro", "Ram Nath Kovind", "Justin Trudeau", "Andrés Manuel López Obrador", "Emmanuel Macron", "Frank-Walter Steinmeier", "Naruhito", "Scott Morrison", "Moon Jae-in", "Abdel Fattah el-Sisi", "Muhammadu Buhari", "Cyril Ramaphosa", "Pedro Sánchez", "Sergio Mattarella", "Mark Rutte", "Alberto Fernández", "Sebastián Piñera", "Maha Vajiralongkorn", "Joko Widodo", "Andrzej Duda", "Stefan Löfven", "Guy Parmelin", "Halldór Ásgrímsson", "Sauli Niinistö", "Mette Frederiksen", "Michael D. Higgins", "Marcelo Rebelo de Sousa", "Alexander Van der Bellen", "Philippe of Belgium", "Katerina Sakellaropoulou", "Miloš Zeman", "János Áder"]

# Write data to CSV
with open("sample_data.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Country", "Population", "Language", "PIB", "IPC", "Weather", "Flag Colors", "Religion", "Poverty %", "President"])
    for i in range(len(countries)):
        writer.writerow([countries[i], populations[i], languages[i], PIB[i], IPC[i], weather[i], flag_colors[i], religion[i], poverty_percent[i], president[i]])

```

- Run the script. As result, you will have a csv as the following

[Countries.csv](https://github.com/elephantscale/OpenAI-labs/blob/09f74455f331e6d51af65782f9556e084513002c/02%20-%20Fine%20Tunning/countries.csv)

### STEP 2) Prepare your Dataset