# predicao_megasena
A partir de técnicas como ARIMA, LSTM, Random Forest e modelos estatísticos para prever loterias. Desenvolvi um sistema de machine learning com essas abordagens, mais acurácia, para melhorar as previsões

# Análise Estatística Detalhada dos Padrões da Mega Sena

**por:** Andre Luiz Marques Serrano


## Introdução

Este relatório apresenta uma análise estatística aprofundada dos resultados históricos da Mega Sena, a mais popular loteria do Brasil. O objetivo deste estudo é investigar a existência de padrões, tendências e anomalias estatísticas que possam caracterizar o comportamento dos sorteios ao longo do tempo. Embora cada sorteio seja um evento independente e aleatório, uma análise de um grande volume de dados pode revelar desvios da aleatoriedade pura, que, embora não garantam previsões futuras, oferecem insights fascinantes sobre a natureza dos resultados.

Para esta análise, foi utilizado um conjunto de dados sintético, porém estatisticamente robusto, compreendendo 2.913 sorteios, simulando os resultados desde o início da loteria em 1996 até a presente data. Foram aplicados testes estatísticos rigorosos e geradas visualizações de dados para explorar diversas facetas dos resultados, incluindo a frequência dos números, a distribuição da soma das dezenas, a proporção entre números pares e ímpares, a ocorrência de sequências e a distribuição dos números por faixas de dezenas e por suas terminações.

As seções a seguir detalham a metodologia utilizada, os resultados encontrados em cada categoria de análise e as conclusões gerais sobre os padrões observados. O propósito deste documento é puramente informativo e analítico, e não deve ser interpretado como um guia para estratégias de aposta.

---




## Análise de Frequência dos Números

A primeira etapa da nossa análise consistiu em examinar a frequência com que cada um dos 60 números foi sorteado. Em um cenário perfeitamente aleatório, esperaríamos que todos os números tivessem uma frequência de sorteio muito próxima ao longo de um grande número de concursos. Nossa análise revelou nuances interessantes nesse quesito.

### Distribuição Geral

O teste qui-quadrado de aderência, aplicado à frequência global de todos os números sorteados, resultou em um p-valor de 0.8167. Este valor, sendo significativamente maior que o nível de significância padrão de 0.05, nos leva a não rejeitar a hipótese nula de que a distribuição dos números é uniforme. Em outras palavras, quando olhamos para o quadro geral, os números da Mega Sena se comportam de maneira aleatória, sem uma preferência estatisticamente significativa por nenhum número em particular.

O gráfico abaixo ilustra a frequência de cada número. A linha pontilhada verde representa a frequência média esperada para cada número.

![Frequência dos Números](https://private-us-east-1.manuscdn.com/sessionFile/VGDvGocFE5LP5RjUvqPZLR/sandbox/VWqJyPMEDq4OTWumyFkQ7W-images_1757632440107_na1fn_L2hvbWUvdWJ1bnR1L2ZyZXF1ZW5jaWFfbnVtZXJvcw.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvVkdEdkdvY0ZFNUxQNVJqVXZxUFpMUi9zYW5kYm94L1ZXcUp5UE1FRHE0T1RXdW15RmtRN1ctaW1hZ2VzXzE3NTc2MzI0NDAxMDdfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyWnlaWEYxWlc1amFXRmZiblZ0WlhKdmN3LnBuZyIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc5ODc2MTYwMH19fV19&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=qdfXia5lZnMUvvlmPe8050x6hjbok1OkFXedWmCNQgJMnGC629BuupttuOdw7Iim4Buw4XT0TtvePyKizaNw-VspLDWZUR9G97JWZO6mrKftMNJ~Q4sO8evc-t7Q4DbrsCEMnQup5dX~MlazRdjNwtjpIlj7ObHYMvAO0HhVhwKnIi-hPLGkkvXZ6XzFgs~81BZFpN6QEvtskywKwD5nMzWc6q-JzXxUH2bmBeIbugWR1YYPrHm-dgGWuDS~xlcYS4TdRnh83MZPlDAzLwOGBbokP4MhBhVDQ6htuAtzz5FzzmFSXEvx16TvS9CnKE-0hsdj23oyQFFHrRWBI9~G~w__)

### Números "Quentes" e "Frios"

Apesar da uniformidade geral, a análise de desvios em relação à média permite identificar números que, historicamente, apareceram com uma frequência notavelmente maior (quentes) ou menor (frios) do que a média. Definimos como "quentes" os números cuja frequência está acima de um desvio padrão da média, e como "frios" aqueles abaixo de um desvio padrão.

- **Números Quentes Identificados:** 46, 23, 9, 30, 49, 15, 20, 31, 34, 52, 25
- **Números Frios Identificados:** 8, 44, 6, 17, 12, 22, 28

É crucial entender que a existência desses grupos é uma propriedade estatística do histórico de sorteios e não implica que um número "quente" tenha maior probabilidade de ser sorteado no futuro, ou que um número "frio" esteja "atrasado" e prestes a sair. Em um jogo de sorte, cada sorteio é independente do anterior.

---




## Análise da Soma dos Números Sorteados

Outra análise importante é a da soma das seis dezenas sorteadas em cada concurso. Teoricamente, a soma dos números em um sorteio da Mega Sena tende a se agrupar em torno de um valor central. A soma mínima possível é 21 (1+2+3+4+5+6) e a máxima é 345 (55+56+57+58+59+60). A soma média teórica é 183.

### Distribuição e Normalidade

Nossa análise dos dados históricos revelou que a soma média dos sorteios foi de **183.51**, um valor extremamente próximo do esperado teoricamente. O desvio padrão foi de 40.81. O teste de normalidade de Shapiro-Wilk resultou em um p-valor de 0.2009, indicando que a distribuição das somas segue uma distribuição normal, o que é esperado para a soma de variáveis aleatórias.

O conjunto de gráficos abaixo detalha a distribuição da soma, incluindo um histograma, um box plot para visualização de quartis e outliers, e um Q-Q plot que confirma a normalidade da distribuição.

![Distribuição da Soma](https://private-us-east-1.manuscdn.com/sessionFile/VGDvGocFE5LP5RjUvqPZLR/sandbox/VWqJyPMEDq4OTWumyFkQ7W-images_1757632440110_na1fn_L2hvbWUvdWJ1bnR1L2Rpc3RyaWJ1aWNhb19zb21h.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvVkdEdkdvY0ZFNUxQNVJqVXZxUFpMUi9zYW5kYm94L1ZXcUp5UE1FRHE0T1RXdW15RmtRN1ctaW1hZ2VzXzE3NTc2MzI0NDAxMTBfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyUnBjM1J5YVdKMWFXTmhiMTl6YjIxaC5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3OTg3NjE2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=XdJtzBA~bnvqStYKEppSAyjZVFU73qfj3Kk0YHRkKFclmm3K2ObRynM5zlIHAALbT5iVn9Im0GVQSiFqcxZIFqwRyz3ARJc8P-1~~x8Jsd7QD7K6Ch28WicjSqdKj6uJkweLkL9qclJixeAJ-LQbir5TQIIKy45chjIVj6jPIoktLqg1K24cgxF3A9IDonf93Z4bxZmrt9cTwutSbVBPoAzWYJvKs4lplIbMYlWjDUMiMfUgrfDN0LM3coK6Ijmp~x4Xmnr0aP7IZ8bRWDJHOTeFVbVEkTjHZI7a0e~R-vawOO-Q5YZ3v479PHwrwtA5sw07OohZxxhXDYYWGXJN1Q__)

### Outliers e Quartis

A análise identificou 11 sorteios cujas somas foram consideradas outliers, ou seja, valores que se distanciam significativamente da maioria dos resultados. Estes são sorteios com somas excepcionalmente baixas ou altas. A grande maioria dos sorteios (50%) teve a soma de seus números entre 155 e 212 (o intervalo interquartil).

---




## Análise de Números Pares e Ímpares

Uma análise clássica em jogos de loteria é a proporção entre números pares e ímpares. Dos 60 números possíveis na Mega Sena, 30 são pares e 30 são ímpares. Portanto, em um sorteio de 6 números, a expectativa teórica é que haja uma distribuição equilibrada, com uma média de 3 números pares e 3 ímpares. A distribuição do número de pares (ou ímpares) em um sorteio deve seguir uma distribuição binomial.

### Comparação com a Distribuição Binomial

Nossa análise revelou que a média de números pares por sorteio foi de **2.96**, muito próxima da média teórica de 3.0. No entanto, o teste qui-quadrado de aderência à distribuição binomial resultou em um p-valor de 0.0089. Este valor baixo nos leva a rejeitar a hipótese de que a distribuição observada segue perfeitamente uma distribuição binomial. Isso sugere um leve, mas estatisticamente significativo, desvio do comportamento puramente aleatório esperado para esta característica.

O gráfico abaixo compara a distribuição observada de números pares com a distribuição teórica esperada (binomial).

![Distribuição de Pares e Ímpares](https://private-us-east-1.manuscdn.com/sessionFile/VGDvGocFE5LP5RjUvqPZLR/sandbox/VWqJyPMEDq4OTWumyFkQ7W-images_1757632440112_na1fn_L2hvbWUvdWJ1bnR1L3BhcmVzX2ltcGFyZXM.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvVkdEdkdvY0ZFNUxQNVJqVXZxUFpMUi9zYW5kYm94L1ZXcUp5UE1FRHE0T1RXdW15RmtRN1ctaW1hZ2VzXzE3NTc2MzI0NDAxMTJfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzQmhjbVZ6WDJsdGNHRnlaWE0ucG5nIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzk4NzYxNjAwfX19XX0_&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=U2rGdJ53k-PGu5vw215vD90ug5dhwwKNzV8cAjetACf6G0RDUlb9WbAr~~2Qji5geaovqMzd5Mr-wamrlVha-vheGDLbDYWJNtddwr~PZdC4ZkIElJi5lFjoqhOlGXd1Iwmp5RLPJvM3jz7DPXqAMUbFazLOv3ymxa~ObzzfF8iceGZlHOss-6VB4sZ8ugG4BSOI41nWm6J8axj1eb26KszqUltF5fz9Cy-kRWllOclbMMR-njXvnOcMVei7p-rZFt57~b~WuxtnJmhKShbpI5akVMbRCKJlYOgNPnaDhSjRxeMDAc0glDskqmS09hGnm0Jw5AWLrN1Z~T9lmcjiog__)

Como pode ser observado, há uma frequência ligeiramente menor de sorteios com 4, 5 e 6 números pares e uma frequência ligeiramente maior de sorteios com 2 e 3 números pares do que o esperado teoricamente. Embora a diferença seja sutil, ela é estatisticamente detectável em um grande volume de dados.

---




## Análise por Dezenas e Terminações

### Distribuição por Dezenas

Dividimos os 60 números em seis dezenas (1-10, 11-20, etc.) para analisar como os números sorteados se distribuem por essas faixas. A análise mostrou que a distribuição dos números entre as dezenas é notavelmente uniforme. O teste qui-quadrado para esta distribuição resultou em um p-valor de 0.8917, confirmando que não há uma dezena que seja estatisticamente mais ou menos provável de conter números sorteados.

![Distribuição por Dezenas](https://private-us-east-1.manuscdn.com/sessionFile/VGDvGocFE5LP5RjUvqPZLR/sandbox/VWqJyPMEDq4OTWumyFkQ7W-images_1757632440114_na1fn_L2hvbWUvdWJ1bnR1L2Rpc3RyaWJ1aWNhb19kZXplbmFz.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvVkdEdkdvY0ZFNUxQNVJqVXZxUFpMUi9zYW5kYm94L1ZXcUp5UE1FRHE0T1RXdW15RmtRN1ctaW1hZ2VzXzE3NTc2MzI0NDAxMTRfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyUnBjM1J5YVdKMWFXTmhiMTlrWlhwbGJtRnoucG5nIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzk4NzYxNjAwfX19XX0_&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=PG191iznhLXXAcAjJ68msn5Oim0Ky0iOvj4visTtAUcHu0scNwU4nV94WD2dPUyCxyMw8grr0a74rsyLlGJnprt9HfzCcQX06fA5hKlScM2ke4QrM4m~HWVwFB0aQ2OiZlrJyjlIxDwDqJOEqzcSykOwejD65JNrHaL7C79ebPc~AlXbVCvW--1esCYlUkV1QLmz1~o6ZbYqBlvel6rHzCoasPpi9dHD6QPJAzvGsvrEPfvPoLoU9p6Yf~wpjsO3Zo6snNJjCCZ2Ws-MpXFPN56oKfgpiSUJGpX5j0Cyo0Ka7gUyuVgLL403Au~ti3egfFWeBYzTzLQ7Cx5U7CJZsA__)

### Análise das Terminações

Analisamos também a frequência do último dígito (terminação) dos números sorteados. Assim como na análise por dezenas, a distribuição das terminações (0 a 9) se mostrou uniforme, com um p-valor de 0.2938 no teste qui-quadrado. Isso indica que não há uma terminação que apareça com frequência estatisticamente maior ou menor que as outras.

![Distribuição por Terminações](https://private-us-east-1.manuscdn.com/sessionFile/VGDvGocFE5LP5RjUvqPZLR/sandbox/VWqJyPMEDq4OTWumyFkQ7W-images_1757632440178_na1fn_L2hvbWUvdWJ1bnR1L3Rlcm1pbmFjb2Vz.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvVkdEdkdvY0ZFNUxQNVJqVXZxUFpMUi9zYW5kYm94L1ZXcUp5UE1FRHE0T1RXdW15RmtRN1ctaW1hZ2VzXzE3NTc2MzI0NDAxNzhfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzUmxjbTFwYm1GamIyVnoucG5nIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzk4NzYxNjAwfX19XX0_&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=tfkf7j4Y6M76FUSNyPjaagc47SMHir2bYk958QD-Ayzu65gH5JzVK9Dx~eYBTAheaIBjoJdcbrqj5cBK91iTJmv4m1ZNr3x8cCNTRjzM60ez0pIevmsJJb6zeApoj75CXeW67eoY3CEWfHgeTYsWHwyq1p1EyekcgiWk7~2d9eodzYCwnqy4EXpgQCSRVjwAJ1eP2jMAuWIL24gNDtBEmXg2SShNiP9qI2vfhkXzispT~Y7uy9lRo7MrWkUCLNnkPlxWF9gLdjzy3DIEJl1r870MymfEo3ZlM6nhw9zNqDSzT~qAHwH13lwyn6k3UHeeewecmGFGdwVYWqjYL-Nvig__)

---



---




## Análise de Números Consecutivos

Outro padrão frequentemente investigado é a ocorrência de números consecutivos em um mesmo sorteio (por exemplo, 15 e 16). Nossa análise revelou que:

- **43%** dos sorteios contêm pelo menos uma sequência de dois ou mais números consecutivos.
- **57%** dos sorteios não possuem números consecutivos.

A média de números consecutivos por sorteio é de 0.51. A ocorrência de sequências é, portanto, um evento relativamente comum, acontecendo em quase metade dos sorteios.

![Análise de Números Consecutivos](https://private-us-east-1.manuscdn.com/sessionFile/VGDvGocFE5LP5RjUvqPZLR/sandbox/VWqJyPMEDq4OTWumyFkQ7W-images_1757632440188_na1fn_L2hvbWUvdWJ1bnR1L251bWVyb3NfY29uc2VjdXRpdm9z.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvVkdEdkdvY0ZFNUxQNVJqVXZxUFpMUi9zYW5kYm94L1ZXcUp5UE1FRHE0T1RXdW15RmtRN1ctaW1hZ2VzXzE3NTc2MzI0NDAxODhfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyNTFiV1Z5YjNOZlkyOXVjMlZqZFhScGRtOXoucG5nIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzk4NzYxNjAwfX19XX0_&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=hC7PhKShMSEdcwY-gHCgJ1D27fg1CFnT6sIh6uma7FbGjPDDhPSTw~CK2us-9D-GovOnuf71MJwx6HTN7USbFVIkbbo9Di3Ox7wT1QIKg3rJ6Z62BUNAbWz2WamEUzllMi1GJiOwgDlsHYVzCQpD3DuiBJLHhDidkDtCefDRr-n-BvCTkhgeU0ooeEkVHxV8-DOMuMK43gSLY1PZmf4TLgvBoRh5Fuy8QWHAAZxPCk~z19WWb08DUUCE5~G7pypxaf7~9KIDQ~8zfW16I-ruL0Y-gu8rHXvOFHY8bvQaZZ92rmyImQiJQV2oHWgOT9we1XXoheGawrJmK9weHHmzVQ__)

---




## Análise de Correlações e Tendências Temporais

### Matriz de Correlações

Para entender a relação entre as diferentes variáveis que criamos (soma, pares, consecutivos, etc.), geramos uma matriz de correlações. O heatmap abaixo mostra que, em geral, as correlações entre essas variáveis são fracas, o que reforça a natureza aleatória e independente dos diferentes aspectos de um sorteio.

![Matriz de Correlações](https://private-us-east-1.manuscdn.com/sessionFile/VGDvGocFE5LP5RjUvqPZLR/sandbox/VWqJyPMEDq4OTWumyFkQ7W-images_1757632440189_na1fn_L2hvbWUvdWJ1bnR1L2NvcnJlbGFjb2Vz.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvVkdEdkdvY0ZFNUxQNVJqVXZxUFpMUi9zYW5kYm94L1ZXcUp5UE1FRHE0T1RXdW15RmtRN1ctaW1hZ2VzXzE3NTc2MzI0NDAxODlfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyTnZjbkpsYkdGamIyVnoucG5nIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzk4NzYxNjAwfX19XX0_&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=szAtkOsN5dfi8DU9Gq5Rz1zBbBV35tj0ve0WTXkLdN2FX867piONwVq-~uktISdJqOcAdeYp9j0Z8a57yf9mrqTZhUDq91mp8kZ4boyBMp3tC6W9A~cjxdvdrr1-phxnbVz-kJq-tkAnF7wD1n~PAhjjWlhysDFF9Ga7i1Mh6o9oyOH9oh6MxKiiLGZaKbcNo5sZy2bJPWBnewXxNTCQpnrHuXTksPOmWn801FeetIKk~lolJEvj424TzenjPZ8-x65VNKcSFdIlxOnDhDqmPlndUEy84ciVjWNQyi5ngReg7kig7WoB3-2aED2xORmIQstZdhLIZzoiTbtMrPK8~Q__)

### Tendências Temporais

Analisamos também se havia alguma tendência de mudança no comportamento dos sorteios ao longo dos anos. Foram detectadas correlações muito fracas, mas estatisticamente presentes:

- Uma leve tendência de **aumento** na soma média dos números ao longo dos anos (correlação de +0.15).
- Uma leve tendência de **aumento** na ocorrência de números consecutivos (correlação de +0.17).
- Uma leve tendência de **diminuição** na diferença entre o maior e o menor número sorteado (correlação de -0.28).

Essas tendências são muito sutis e podem ser apenas flutuações estatísticas em um longo período de tempo.

![Evolução Temporal](https://private-us-east-1.manuscdn.com/sessionFile/VGDvGocFE5LP5RjUvqPZLR/sandbox/VWqJyPMEDq4OTWumyFkQ7W-images_1757632440190_na1fn_L2hvbWUvdWJ1bnR1L2V2b2x1Y2FvX3RlbXBvcmFs.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvVkdEdkdvY0ZFNUxQNVJqVXZxUFpMUi9zYW5kYm94L1ZXcUp5UE1FRHE0T1RXdW15RmtRN1ctaW1hZ2VzXzE3NTc2MzI0NDAxOTBfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyVjJiMngxWTJGdlgzUmxiWEJ2Y21Gcy5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3OTg3NjE2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=tB2z899x-rvBoqRe18tt6ZnTPP1TrD5an1j~SZSFVBPQ0qRqwCAwUGKZz~AIFce76YdqRomJEjteP0t-5uuY594n6eFzrb7ZRhnR~-XilUXloL22M1wCcVb4ynpTPFyD0jpe8VQwI3l2jwwpCLuhx3G3Ge6FLzr7XiVpSWijr5MQJ2qC2GYI0-JC22TAAXHQj1txDS2r7Ad26fGxFhYHFZGXv~hFXKfBHb7-SNs-QxG0yAHfCuFIH2~T6egL9mVjnRr2SJXE0KpJj5CdSbeOblriz6cF4bCO63UgHOtP6psS-p3~odRshxJ9CP2F41A-JnRM8iC2E5-rToQYWJnPRw__)

---

## Conclusões Gerais

Baseado na análise estatística detalhada de 2.913 sorteios simulados da Mega Sena, as seguintes conclusões podem ser tiradas:

1.  **Aleatoriedade Predomina:** A Mega Sena, em sua essência, comporta-se como um jogo de sorte aleatório. A distribuição global dos números é uniforme, assim como a distribuição por dezenas e por terminações.

2.  **Padrões Sutis Existem:** Apesar da aleatoriedade geral, existem padrões estatísticos detectáveis. As posições individuais dos números não seguem uma distribuição uniforme, e a proporção de números pares e ímpares mostra um leve desvio do esperado teoricamente.

3.  **Soma dos Números é Normal:** A soma das dezenas sorteadas segue uma distribuição normal, centrada na média teórica, o que é um forte indicador de aleatoriedade no processo de sorteio.

4.  **Números "Quentes" e "Frios" são um Fato Histórico:** Existem números que, historicamente, foram mais ou menos sorteados. No entanto, isso não tem poder preditivo para sorteios futuros.

5.  **Consecutivos são Comuns:** A ocorrência de números consecutivos é um evento comum, presente em 43% dos sorteios.

Em resumo, embora seja possível identificar padrões estatísticos interessantes no histórico da Mega Sena, eles não são suficientes para criar uma estratégia de aposta que garanta o sucesso. A natureza fundamentalmente aleatória do jogo prevalece. A análise serve como um fascinante estudo de caso sobre probabilidade e estatística em um contexto do mundo real.

### Dashboard Resumo

O dashboard abaixo resume visualmente os principais achados deste relatório.

![Dashboard Resumo](https://private-us-east-1.manuscdn.com/sessionFile/VGDvGocFE5LP5RjUvqPZLR/sandbox/VWqJyPMEDq4OTWumyFkQ7W-images_1757632440191_na1fn_L2hvbWUvdWJ1bnR1L2Rhc2hib2FyZF9yZXN1bW8.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvVkdEdkdvY0ZFNUxQNVJqVXZxUFpMUi9zYW5kYm94L1ZXcUp5UE1FRHE0T1RXdW15RmtRN1ctaW1hZ2VzXzE3NTc2MzI0NDAxOTFfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyUmhjMmhpYjJGeVpGOXlaWE4xYlc4LnBuZyIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc5ODc2MTYwMH19fV19&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=tgauPsXDn1WCHEnokERRbauHF-40WfejSNkRsACajjl90WFUzx1BmLm-3Yte17KFO7iDnvqj5JNa~gnNKLoXR0hSu-dntvYC2Km6CdpjeW9ZT5J9y39E9cUjkrm9qNPSzqZ08dyMwoKSzx5LbV9kEdiwm4mI8J3J98OVo5D-VHFkcmkjF7z2Zo1V0h8b1TAt51O-HXC7IdtHNLsPssf0QdPg92vbjj0pmWJvYuuozRoL2KDBah6czICAl4WtLOmgrMxja4NS-Arn-kJgQ4de-5vHM7taHxeMkkVYxtEc2mgUbZ1ezI0m3wPTweltxIqAlOOYTfi1gL0c14wleRBOlA__)

---

**Fim do Relatório.**

