import os
import numpy as np
import pandas as pd

# ————— Parámetros —————
SERIES_FILE    = 'salidastxt/nubes_multilayer.txt'
OUT_ROOT       = 'salidastxt/estadisticasnubes'
PX_TO_M = {2: 119.5, 3: 179.3, 4: 239.1}
FIXED_MARGIN_M = 475.0
RAIN_THRESHOLD = 476.0

# ————— Crear carpetas de salida —————
for layer_case in ('all', 'no_multilayer', 'allall'):
    for rain_case in ('all', 'no_rain'):
        d = os.path.join(OUT_ROOT, layer_case, rain_case)
        os.makedirs(d, exist_ok=True)

# ————— 1) Leer el TXT multilayer —————
# Columnas: time_index, layer1, layer2, layer3, cloud_base_height_m
df = pd.read_csv(
    SERIES_FILE, sep=r'\s+', parse_dates=['time_index']
)

# Extraer arrays
truth_array = df[['layer1','layer2','layer3']].values
pred_array  = df['cbase'].values

# Construir listas
truth_layers = [list(filter(lambda x: not np.isnan(x), row))
                for row in truth_array]
multi_flags   = [len(l) > 1 for l in truth_layers]
rain_flags    = [ (not np.isnan(p) and p <= RAIN_THRESHOLD)
                  for p in pred_array ]

# ————— 2) Funciones de confusión  —————
def confusion_single(idxs, margin):
    TP = FP = FN = TN = 0
    for i in idxs:
        truths = truth_layers[i]
        t = truths[0] if truths else np.nan
        p = pred_array[i]
        true_c = not np.isnan(t)
        pred_c = not np.isnan(p)

        if true_c and pred_c:
            d = abs(p - t)
            if d <= margin:
                TP += 1
            else:
                FP += 1
        elif true_c and not pred_c:
            FN += 1
        elif not true_c and pred_c:
            FP += 1
        else:
            TN += 1

    tot  = TP + FP + FN + TN
    acc  = 100 * (TP + TN) / tot      if tot else 0
    prec = 100 * TP       / (TP + FP) if (TP + FP) else 0
    rec  = 100 * TP       / (TP + FN) if (TP + FN) else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return TP, FP, FN, TN, tot, acc, prec, rec, f1


def confusion_any(idxs, margin):
    TP = FP = FN = TN = 0
    for i in idxs:
        truths = truth_layers[i]
        p = pred_array[i]
        true_c = bool(truths)
        pred_c = not np.isnan(p)

        if true_c and pred_c:
            # al menos una capa predice dentro del margen
            if any(abs(p - t) <= margin for t in truths):
                TP += 1
            else:
                FP += 1
        elif true_c and not pred_c:
            FN += 1
        elif not true_c and pred_c:
            FP += 1
        else:
            TN += 1

    tot  = TP + FP + FN + TN
    acc  = 100 * (TP + TN) / tot      if tot else 0
    prec = 100 * TP       / (TP + FP) if (TP + FP) else 0
    rec  = 100 * TP       / (TP + FN) if (TP + FN) else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return TP, FP, FN, TN, tot, acc, prec, rec, f1

# ————— 3) Escenarios y generación de TXT —————
scenarios = [
    # (folder, multilayer_req, rain_case, use_any)
    ('all',         None,   'all',     False),
    ('all',         None,   'no_rain', False),
    ('no_multilayer', False, 'all',    False),
    ('no_multilayer', False, 'no_rain',False),
    ('allall',      None,   'all',     True),
    ('allall',      None,   'no_rain', True),
]

for layer_case, multilayer_req, rain_case, use_any in scenarios:
    idxs = []
    for i in range(len(df)):
        ok_layer = (multilayer_req is None) or (multi_flags[i] == multilayer_req)
        ok_rain  = (rain_case=='all') or (not rain_flags[i])
        if ok_layer and ok_rain:
            idxs.append(i)
    out_dir = os.path.join(OUT_ROOT, layer_case, rain_case)

    # márgenes en píxels
    for px, margin in PX_TO_M.items():
        func = confusion_any if use_any else confusion_single
        TP, FP, FN, TN, tot, acc, prec, rec, f1 = func(idxs, margin)
        fn = os.path.join(out_dir, f'confusion_{px}px.txt')
        with open(fn, 'w') as f:
            f.write(f"Confusión — {layer_case} — ±{px} px (±{margin:.1f} m)\n\n")
            f.write(f"TP: {TP}\nFP: {FP}\nFN: {FN}\nTN: {TN}\n\n")
            f.write(f"Total: {tot}\n")
            f.write(f"Accuracy: {acc:.2f} %\n")
            f.write(f"Precision: {prec:.2f} %\n")
            f.write(f"Recall: {rec:.2f} %\n")
            f.write(f"F1 score: {f1:.2f} %\n")

    # margen fijo 475 m
    func = confusion_any if use_any else confusion_single
    TP, FP, FN, TN, tot, acc, prec, rec, f1 = func(idxs, FIXED_MARGIN_M)
    fn = os.path.join(out_dir, 'confusion_475m.txt')
    with open(fn, 'w') as f:
        f.write(f"Confusión — {layer_case} — ±{int(FIXED_MARGIN_M)} m\n\n")
        f.write(f"TP: {TP}\nFP: {FP}\nFN: {FN}\nTN: {TN}\n\n")
        f.write(f"Total: {tot}\n")
        f.write(f"Accuracy: {acc:.2f} %\n")
        f.write(f"Precision: {prec:.2f} %\n")
        f.write(f"Recall: {rec:.2f} %\n")
        f.write(f"F1 score: {f1:.2f} %\n")

print(" Estadísticas de confusión generadas para ALL, NO_MULTILAYER y ALLALL.")



