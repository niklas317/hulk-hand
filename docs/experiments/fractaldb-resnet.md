# FractalDB-ResNet-Experimente

## 1. Ausgangssituation

Der private NEW-Datensatz umfasst 13.000 offline vorverarbeitete Bilder aus 26
Aufnahmesessions. Die vier Klassen und ihre feste Reihenfolge sind:

```text
0 gesture_1
1 gesture_2
2 gesture_3
3 no_gesture
```

Pro Session liegen 100 Bilder je Geste und 200 Bilder für `no_gesture` vor.
`no_gesture` ist deshalb keine einfache Hintergrundklasse: Sie umfasst Bilder
ohne Hand, unbekannte Gesten und Aufnahmen mit mehreren Händen. Die Bilder
wurden als Burst-Aufnahmen mit unterschiedlichen Handpositionen und
Kameraabständen aufgenommen. Zwei Kameras, wechselnde Hintergründe und
sessionabhängige Bildkompositionen erzeugen einen deutlichen
Domänenunterschied.

Die Aufteilung wurde durchgehend beibehalten:

```text
A-R: Training (18 Sessions, 9.000 Bilder)
S-V: Validation (4 Sessions, 2.000 Bilder)
W-Z: Test (4 Sessions, 2.000 Bilder)
```

Die Eingabebilder sind offline auf 224x224 vorverarbeitet. Es wurden keine
Handdetektion, kein Cropping und kein Entfernen von Bildern verwendet.

## 2. Experimente und Ergebnisse

Die folgenden Werte stammen aus vorhandenen Berichten, Checkpoint-Metadaten
und den festgehaltenen Versuchsergebnissen. Werte mit `~` sind ausdrücklich
Näherungen; fehlende Metriken wurden nicht nachträglich erfunden.

### 2.1 ResNet ohne Pretraining

- Architektur: ResNet-basierter privater NEW-Klassifikator ohne externes
  Pretraining.
- Initialisierung: zufällig.
- Ergebnis: ungefähr 30% Validation Accuracy.
- Schlussfolgerung: Die Architektur kann die Trainingsdaten anpassen, erhält
  aber ohne brauchbare visuelle Initialisierung keine robuste Session-
  Generalisierung.

### 2.2 FractalDB-ResNet34, reguläres Fine-Tuning

- Architektur: torchvision ResNet34.
- Initialisierung: `totti0223/resnet34_fractaldb_10000`.
- Preprocessing: ImageNet-Normalisierung und aspect-ratio-erhaltendes
  FractalDB-Letterboxing.
- Training: AdamW, session-balanced Batches, Warmup/Cosine-Scheduler,
  Label Smoothing und Gradient Clipping.
- Ergebnis: ungefähr 40-45% Validation Accuracy.
- Beobachtung: deutlich besser als zufällige Initialisierung, aber weit von
  einer sessionrobusten Lösung entfernt.

### 2.3 E1 – korrigierte Augmentation

- Eine konservative `RandomAffine` ersetzte die Kombination aus Rotation und
  Affine-Transformation.
- Parameter: `degrees=12`, `translate=(0.06, 0.06)`, `scale=(0.94, 1.06)`,
  `shear=3`, bilineare Interpolation.
- `RandomErasing` wurde deaktiviert; ColorJitter blieb erhalten.
- Das Training nutzte weiterhin die damalige staged Fine-Tuning-Konfiguration.
- Ein belastbarer, separater E1-Endwert ist in den verfügbaren Artefakten nicht
  dokumentiert; die Änderung ist daher nicht mit einer erfundenen Accuracy zu
  bewerten.

### 2.4 E2 – konservatives Fine-Tuning

- Backbone Learning Rate: `1e-5`.
- Classifier Learning Rate: `3e-4`.
- Label Smoothing: `0.05`.
- Early-Stopping-Patience: `7`.
- Epochen 1-3: nur Classifier; ab Epoche 4: `layer4` plus Classifier.
- `layer1` blieb eingefroren.
- Ein separater, reproduzierbarer Endwert ist in den verfügbaren Logs nicht
  vollständig erhalten. Bekannte ResNet34-Referenzläufe lagen weiterhin bei
  ungefähr 40-45% Validation Accuracy.

### 2.5 E3 – BatchNorm-Experiment

- E2-Konfiguration plus BatchNorm2d-Module während des Trainings in `eval()`.
- Ziel: Verhindern, dass kleine bzw. sessionspezifische Batches die laufenden
  BatchNorm-Statistiken verschieben.
- Ein separater valider Endwert ist nicht dokumentiert.

### 2.6 Burst-Aufnahmen, Redundanz und Bildqualität

Die Burst-Struktur erklärt, warum hohe Trainingswerte nicht automatisch neue
Sessions repräsentieren. Benachbarte Bilder können sehr ähnliche Handposen,
Kameradistanzen und Hintergründe teilen. Ein Modell kann dadurch Aufnahme-
oder Session-Merkmale statt allgemein übertragbarer Gestenmerkmale lernen.

Nachgewiesen ist die starke Sessionabhängigkeit der Accuracy. Die genaue
Wirkung einzelner Burst-Deduplikations- oder Qualitätsfilter ist in den
vorliegenden Artefakten nicht als kontrollierter, separater Versuch
quantifiziert; diese Punkte bleiben daher Hypothesen bzw. Diagnoseansätze.

### 2.7 Knowledge Distillation v1

- Teacher: fine-getuntes DINOv2 ViT-S/14 mit 384-D-Embedding und vier Klassen.
- Student: FractalDB-ResNet34 mit 512-D-Embedding.
- Teacher vollständig eingefroren; keine Teacher-Gewichte im Deployment-
  Modell.
- Gemeinsame Bildaugmentation, getrennte ImageNet-kompatible Normalisierung.
- Loss: zunächst 50% Cross-Entropy und 50% Logit-KD bei `T=2.0`.
- Beobachtung: Der Teacher erreichte ungefähr 89% Validation Accuracy,
  während der Student deutlich schwächer blieb. Ein früher Lauf lag bei
  31,95% Validation Accuracy; weitere Distillation-Läufe lagen ungefähr bei
  40-42%.
- Der Student konnte Trainingsdaten nahezu perfekt an den Teacher anpassen,
  generalisierte aber nicht auf S-V.

### 2.8 Knowledge Distillation v2 / E5 v2

- Progressive Stages: Classifier-only in Epoche 1, `layer4+classifier` in
  Epochen 2-3, ab Epoche 4 gesamter trainierbarer Backbone außer `layer1`.
- Layerabhängige Learning Rates: Stem `1e-5`, `layer2` `2e-5`, `layer3`
  `3e-5`, `layer4` `5e-5`, Classifier `3e-4`.
- Loss: 70% Cross-Entropy, 30% Logit-KD, `T=2.0`, Label Smoothing `0.05`.
- Teacher-Überprüfung: ungefähr 89% Validation Accuracy und ungefähr 98,4%
  Accuracy auf augmentierten Trainingsbildern.
- Teacher-Student-Agreement auf Validation: ungefähr 39,8%.
- Student-Training: bis ungefähr 99,5% Accuracy möglich, ohne entsprechende
  Validation-Generaliserung.
- Diese Werte zeigen ein Overfitting-Problem und keinen Nachweis, dass die
  stärkere Unfreezing-Strategie die unbekannten Sessions verbessert.

### 2.9 Abschließende Teacher-Student-Validation-Diagnose

Die Diagnose verglich Teacher und Student auf exakt denselben S-V-Dateien und
berichtete Agreement, KL-Divergenz, Klassen-/Sessionwerte sowie
Teacher-korrekte/Student-falsche Beispiele. Der zentrale Befund war das
niedrige Agreement von ungefähr 39,8% trotz des starken Teachers. Besonders
relevant waren Fälle, in denen eine echte Geste korrekt durch den Teacher und
als `no_gesture` durch den Student erkannt wurde.

## 3. Fehleranalyse

### Nachgewiesene Befunde

- Große Generalisierungslücke: Training bis ungefähr 99,5%, Validation beim
  Distillation-Studenten nur ungefähr 40-42%.
- Starke Abhängigkeit von den Aufnahmesessions.
- `no_gesture` wird häufig vorhergesagt; dadurch kann ein Modell bei Gesten
  scheinbar plausible Gesamtwerte erreichen, obwohl die Gesten-Recalls niedrig
  bleiben.
- Die Validation-Verteilung beträgt 20/20/20/40% für die drei Gesten und
  `no_gesture`. Ein reiner Mehrheitsklassen-Klassifikator erreicht daher 40%
  Accuracy.
- Geringe Balanced Accuracy der FractalDB-Studenten zeigt, dass die
  Gesamtaccuracy die schwachen Gestenklassen verdecken kann.
- Distillation überträgt das Trainingssignal gut auf A-R, aber nicht robust auf
  neue Sessionbedingungen S-V.

### Hypothesen

- Burst-Redundanz und sessiongebundene Hintergründe fördern das Lernen von
  Aufnahmekontext statt Gestenform.
- Das FractalDB-Pretraining stellt für diese natürliche Gesten-/Kamera-
  Domäne möglicherweise weniger übertragbare Features bereit als ein
  natürliches ImageNet-Pretraining.
- Die starke `no_gesture`-Klasse kann die Entscheidungsgrenze konservativ in
  Richtung `no_gesture` verschieben.

Diese Hypothesen sind plausible Erklärungen, aber nicht jeweils durch einen
kontrollierten Einzelversuch bewiesen.

## 4. Fazit

Weitere FractalDB-ResNet-Experimente werden vorerst eingestellt. Das bisherige
FractalDB-Pretraining liefert für diese Aufgabe nicht genügend nachweisbar
übertragbare Features, um die Sessionverschiebung zuverlässig zu überwinden.

Das bedeutet nicht, dass ResNet grundsätzlich ungeeignet ist. Als nächster
kontrollierter Vergleich wird ein unverändertes ResNet18 mit natürlichem
ImageNet-1K-Pretraining untersucht. Dadurch wird der Einfluss der
Initialisierung getrennt von Architektur, Dataset-Split und Evaluationslogik
getestet.
