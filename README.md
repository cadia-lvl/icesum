# IceSum
IceSum is a neural network-based extractive summarization tool for Icelandic news text. It was trained on a dataset
of 1,000 Icelandic news articles which were manually annotated with extractive summaries. The models were trained
using the [nnsum](https://github.com/kedz/nnsum) library.

## Installation
```
git clone https://github.com/jonfd/icesum.git
cd icesum
pip install torch==0.4.1 -f https://download.pytorch.org/whl/cpu/torch_stable.html
pip install pytorch-ignite==0.3.0
pip install tokenizer
git clone https://github.com/kedz/nnsum.git
cd nnsum
python setup.py install
```

## Models
The following models are available for download:
* [mbl-cnn-s2s.pth](https://www.dropbox.com/s/m74x0vcenvow7i3/mbl-cnn-s2s.pth?dl=0): A CNN encoder with a sequence to sequence extractor.

More models are forthcoming.

## Data
The dataset will be released with an open license in November, 2020.

## Demo
An online demo will be available shortly.

## Example
```python
from icesum import Summarizer
summarizer = Summarizer('models/mbl-cnn-s2s.pth')

# Source: https://www.ruv.is/frett/2020/10/30/domstolar-nyta-ekki-fjarfundarbunad-sem-skyldi
article = """
Ákærendafélag Íslands segir að dómstólar séu of illa búnir til að geta nýtt sér
bráðabirgðaheimild til málsmeðferða í gegnum fjarfundarbúnað. Umsýsla dómstóla
segir að tæknilausnir séu til staðar en að unnið sé að úrbótum.

Þetta kemur fram á forsíðu Fréttablaðsins í morgun. Í vor var sett í lög
bráðabirgðaákvæði þess efnis að heimilt sé að skýrslutaka og önnur málsmeðferð
fari fram í gegnum fjarfundarbúnað. Var það gert í sóttvarnarskyni vegna
faraldursins. Dómsmálaráðherra lagði nýverið fram frumvarp um framlengingu
þessarar heimildar sem nær út árið 2021 en hún féll úr gildi 1. október. 

Í umsögn Ákærendafélags Íslands kemur fram að dómstólar hafi ekki nýtt sér
þessa heimild eins og kostur er bestur í ljósi þess tækjabúnaður sé ekki
fullnægjandi. Þá sé allur gangur á því hvaða kröfur séu gerðar til þeirra sem
gefa skýrslu varðandi staðsetningu og einrúm skýrslugjafans. 

„Í sumum tilvikum hefur skýrslugjafi hreinlega staðið úti á götu í heimabæ
sínum en í öðrum tilvikum hafa dómarar verið strangir á því að skýrslugjafi sé á
tilteknum stað og í einrúmi við skýrslugjöfina. Það væri e.t.v. rétt að
lagaákvæðið eða greinargerðin gæfu einhverjar leiðbeiningar í þessum efnum.“
segir í umsögn félagsins.

Einnig kemur fram í umsögninni að heimildin sé af hinu góða þar sem málsmeðferð
hefði tafist verulega ef hennar nyti ekki við. 

„Sumir dómstólar hafa þó lagt sig fram við að bæta tækjabúnað sinn og er það
reynsla ákærenda að í þeim tilvikum þar sem þetta hefur verið framkvæmt hafi það
gefist vel og ef þessarar heimildar hefði ekki notið við væru þessi mál enn til
meðferðar fyrir dómstólum.

Ólöf Finnsdóttir, framkvæmdastjóri dómstólasýslunnar segir við Fréttablaðið að
tækjabúnaður sé til staðar hjá dómstólum til að nýta heimildina. Vandinn sé sá
að tryggja þurfi að allt sem fram fer sé tekið upp í hljóði og mynd. Aðeins
Héraðsdómur Reykjavíkur og Landsréttur séu komnir með slíkt heilstætt kerfi.
Unnið sé að því að koma því við hjá öðrum dómstólum landsins.Það verði gert á
næstu vikum og mánuðum. Þeir hafi þurft að reiða sig á bráðabirgðalausn.
Upptökukerfi í dómsal taki upp það sem fer fram á fjarfundinum.
"""

summary = summarizer.predict(article, summary_length=75)
print(summary)

# Output:
# Ákærendafélag Íslands segir að dómstólar séu of illa búnir til að geta nýtt
# sér bráðabirgðaheimild til málsmeðferða í gegnum fjarfundarbúnað. Umsýsla
# dómstóla segir að tæknilausnir séu til staðar en að unnið sé að úrbótum. Í vor
# var sett í lög bráðabirgðaákvæði þess efnis að heimilt sé að skýrslutaka og
# önnur málsmeðferð fari fram í gegnum fjarfundarbúnað. Í umsögn Ákærendafélags
# Íslands kemur fram að dómstólar hafi ekki nýtt sér þessa heimild eins og
# kostur er bestur í ljósi þess tækjabúnaður sé ekki fullnægjandi.
```

## Requirements
* Python 3.6 - 3.7
* PyTorch 0.4.1
* PyTorch-Ignite 0.3.0
* tokenizer
* [nnsum](https://github.com/kedz/nnsum)

## Contributors
Jón Friðrik Daðason, [Hrafn Loftsson](http://www.ru.is/kennarar/hrafn/), Salome Lilja Sigurðardóttir and Þorsteinn Björnsson contributed to this project.

## Acknowledgments
This project was funded by the Strategic Research and Development Programme for Language Technology 
([Markáætlun í tungu og tækni](https://www.rannis.is/sjodir/rannsoknir/markaaetlun-i-tungu-og-taekni/markaaetlun-i-tungu-og-taekni)).

## License
    Copyright © 2020 Jón Friðrik Daðason

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
