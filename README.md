# NeuroICU-qEEG

Monitor de EEG quantitativo e prognóstico neurológico para pacientes comatosos na UTI.

> **Projeto educacional e de pesquisa. Não usar para decisões clínicas.**

> Em construção: etapa 1 (setup e leitura dos dados). Demo e resultados virão nas próximas etapas.

## Problema

Após uma parada cardíaca, muitos pacientes chegam à UTI em coma, e a decisão de manter ou retirar o suporte depende do prognóstico neurológico. O EEG contínuo ajuda nessa avaliação, mas sua leitura visual é demorada e exige especialistas. Este projeto calcula métricas de EEG quantitativo (qEEG) e estima o desfecho neurológico (CPC 1–2 vs. 3–5) a partir das primeiras horas de registro.

## Dados

O projeto usa o [I-CARE](https://physionet.org/content/i-care/) (PhysioNet), com EEG contínuo de pacientes comatosos pós-parada cardíaca de sete hospitais dos EUA e Europa.

- Os dados **não** estão neste repositório. Licença do dataset: CC BY-NC-SA 4.0.
- O dataset completo tem cerca de 1,5 TB. Para desenvolver, baixe alguns pacientes e só as primeiras horas.
- O código lê qualquer pasta local: formato WFDB (`.hea` + `.mat`) do I-CARE e também EDF/BDF.

Estrutura esperada (a do próprio PhysioNet):

```
training/
└── 0284/
    ├── 0284.txt               # dados clínicos e desfecho
    ├── 0284_001_004_EEG.hea   # segmento 001, hora 004 após a parada
    └── 0284_001_004_EEG.mat
```

## Como rodar

```bash
pip install -e ".[dev]"
cp .env.example .env            # edite NEUROICU_DATA_DIR com o caminho da sua pasta
make index DATA_DIR=/caminho/para/i-care/training
```

`make index` gera `data/processed/index.csv`, com um registro por linha e os dados clínicos de cada paciente.

Em Python:

```python
from neuroicu.io import discover, load_recording, to_bipolar

recs = discover("/caminho/para/i-care/training", max_hour=72)
rec = load_recording(recs[0])          # sinal em µV, formato (canais, amostras)
bip = to_bipolar(rec, [("Fp1", "F7"), ("F7", "T3")])
```

## Estrutura

```
src/neuroicu/   código reutilizável (leitura, pré-processamento, features, modelos)
configs/        bandas, montagem bipolar, janelas, hiperparâmetros
notebooks/      exploração
tests/          testes com dados sintéticos (nenhum dado real)
```

## Origem

Metodologia baseada no meu TCC em Engenharia Biomédica (UFU), sobre processamento quantitativo de EEG em pacientes de UTI, aplicada aqui a dados públicos.

## Citação

Se usar este código, cite o I-CARE:

Amorim, E. et al. (2023). I-CARE: International Cardiac Arrest REsearch consortium Database. PhysioNet. https://doi.org/10.13026/avek-0p97

## Licença

Código sob licença MIT. Os dados do I-CARE seguem a licença própria (CC BY-NC-SA 4.0).
