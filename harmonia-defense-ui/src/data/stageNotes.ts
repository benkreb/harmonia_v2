import type { StageNote } from "./types";

export const stageNotes: Record<number, StageNote> = {
  0: {
    title: "Title",
    targetSec: 30,
    note: "Merhaba. Ben Berk. Bugün size yaklaşık 10 dakikalık bir sunumla Harmonia Vision'u, gizlilik-koruyan federated learning ile meme kanseri segmentasyonunu anlatacağım.",
  },
  1: {
    title: "Problem",
    targetSec: 60,
    note: "Meme kanseri taramasında AI çok güçlü ama hastaneler hasta görüntülerini doğrudan paylaşamıyor. Sorumuz: ham veri hastanede kalırken ortak model eğitilebilir mi?",
  },
  2: {
    title: "Approach",
    targetSec: 75,
    note: "Yaklaşım üç katmanlı: ConnectedUnet + EfficientNet-B3 modeli, Flower FedAvg federated learning, ve TenSEAL CKKS ile şifreli parametre agregasyonu.",
  },
  3: {
    title: "Headline",
    targetSec: 60,
    note: "Ana sonuç: federated learning centralized eğitime çok yakın. IID FL ile centralized Dice farkı yaklaşık 0.003; Non-IID koşulda da fark çok küçük.",
  },
  4: {
    title: "Drilldown",
    targetSec: 90,
    note: "Bu tabloda tüm metrikleri görüyoruz. Dice ve IoU neredeyse aynı; sensitivity tarafında küçük ama dürüstçe not edilmesi gereken bir düşüş var.",
  },
  5: {
    title: "Interactive Demo",
    targetSec: 120,
    note: "Şimdi gerçek mammogram örneklerinde 5 konfigürasyonun tahminlerini yan yana göstereceğim: kolay, tipik ve zor vakalar.",
  },
  6: {
    title: "Limitations",
    targetSec: 60,
    note: "Çalışmanın sınırları: tek seed, local sequential fallback nedeniyle byte overhead artefact'i, ve CKKS secret key'in server tarafında tutulması.",
  },
  7: {
    title: "Conclusion",
    targetSec: 45,
    note: "Sonuç: federated learning clinical parity sağlıyor, Non-IID dağılıma dayanıklı, PHE ise yaklaşık yüzde 25 zaman maliyetiyle Dice kaybı oluşturmuyor.",
  },
};

export const totalTargetSec = Object.values(stageNotes).reduce((acc, stage) => acc + stage.targetSec, 0);
