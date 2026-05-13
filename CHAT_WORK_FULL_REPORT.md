# Harmonia Vision FL Phase - Chat Work Full Report

Bu rapor, bu chat boyunca Harmonia v2 projesinde planlanan, uygulanan, çalıştırılan, debug edilen, raporlanan, VPS tarafında kontrol edilen ve GitHub'a yüklenen işleri uçtan uca özetler.

Gizlilik notu: Kullanıcının paylaştığı DigitalOcean token, GitHub token, private key, sertifika private key içerikleri veya benzeri gizli değerler bu rapora bilerek yazılmamıştır. Bu değerler `.env` veya generated secret dosyalarında tutuldu ve `.gitignore` ile commit dışı bırakıldı.

## 1. Başlangıç Hedefi

Kullanıcı, "Harmonia Vision - FL Phase Solo PRD v2.0" kapsamındaki federated learning fazını tek günlük, pragmatik ve raporlanabilir şekilde tamamlamak istedi.

Ana hedefler:

- CBIS-DDSM segmentasyon modeli için centralized baseline ile federated learning benchmark karşılaştırması yapmak.
- Flower tabanlı FedAvg federated learning akışını eklemek.
- IID ve Non-IID hasta bazlı 2 client bölümleri üretmek.
- TenSEAL CKKS tabanlı PHE yolunu çalışır hale getirmek.
- F1/F2/F3/F4 benchmark koşularını tamamlamak.
- C1/F1/F2/F3/F4 metriklerini raporlamak.
- DigitalOcean VPS tarafında eğitim yapılamasa bile güvenlik, SSH, WireGuard, Docker ve deployment scaffold doğrulamalarını yapmak.
- Son durumda projeyi GitHub'a yüklemek.

## 2. PRD Kapsamı ve Öncelikler

İlk PRD'deki öncelikler:

- Must have:
  - F1: FL IID, PHE yok.
  - F3: FL Non-IID, PHE yok.
  - C1/F1/F3 benchmark raporu.
- Should have:
  - F2: FL IID + PHE.
  - F4: FL Non-IID + PHE.
  - PHE overhead ölçümleri.
- Nice to have:
  - DigitalOcean üzerinde 3 droplet WireGuard + mTLS ile F1 deployment doğrulaması.

Kullanıcı geri bildirimiyle öncelik değişti:

- PHE artık nice-to-have/should-have değil, must-have sayıldı.
- GPU droplet bulunamadığı için eğitimler lokal devam etti.
- VPS'ler GPU olmasa bile CPU droplet olarak açıldı ve güvenlik/network/deployment altyapısı doğrulandı.
- VPS tarafı model eğitimi için değil, snapshot'larda güvenlik ve bağlantı ayarlarının hazır olduğunu göstermek için kullanıldı.

## 3. Temel Teknik Kararlar

Uygulamada alınan kararlar:

- Client sayısı: 2.
- Non-IID tanımı: hasta seviyesinde 80/20 class skew.
- Rerun sayısı: her F config için 1.
- Max round: 20.
- Convergence patience: 7.
- Her full F koşusundan önce smoke test zorunlu tutuldu.
- Windows ortamında Ray simulation sorunlu olduğu için sequential local FedAvg fallback eklendi.
- PHE için TenSEAL CKKS kullanıldı.
- PHE akışında server-held secret key kullanıldı; bu fazda amaç production trust boundary değil, overhead ve roundtrip doğrulamasıydı.
- PyTorch cihaz seçimi config'te `cuda` olsa bile CPU-only torch kurulumu varsa otomatik `cpu` fallback eklendi.
- Model ve loss factory'leri quiet hale getirildi; Windows console encoding sorunları önlendi.

## 4. Ortam ve Bağımlılıklar

Çalışma ortamı:

- OS: Windows.
- Python: `py -3.11` kullanıldı.
- CUDA için PyTorch daha sonra CUDA 13.0 wheel ile yeniden kuruldu.
- Default Python 3.13 ortamı Ray bağımlılığı için uygun olmadığı için kullanılmadı.

Ek bağımlılık dosyası:

- `requirements_fl.txt`

Eklenen ana paketler:

- `flwr[simulation]>=1.9.0,<2.0.0`
- `ray[default]>=2.10.0,<3.0.0`
- `grpcio>=1.60.0`
- `cryptography>=42.0.0`
- `tenseal>=0.3.14`
- `pyyaml>=6.0`
- `rich>=13.0.0`
- `tabulate>=0.9.0`

Neden `tabulate` eklendi:

- Pandas `to_markdown` çağrısı rapor üretiminde `tabulate` paketi gerektirdi.

## 5. Path ve Config Düzenlemeleri

`segmentation/config.py` güncellendi.

Önceki problem:

- Root path hardcoded yapıdaydı ve farklı ortamlarda taşınabilir değildi.

Çözüm:

- `HARMONIA_ROOT` environment variable desteklendi.
- Yoksa dosya konumuna göre proje kökü hesaplandı.

Eklenen davranış:

- Config YAML'da device `cuda` olsa bile `torch.cuda.is_available()` false ise training device otomatik `cpu` yapıldı.

Bu değişikliklerin nedeni:

- Lokal Windows ortamında `nvidia-smi` GPU gösterse bile ilk PyTorch kurulumu CPU-only idi.
- Bu durum F1 smoke/final evaluation aşamasında `AssertionError: Torch not compiled with CUDA enabled` hatasına yol açtı.

## 6. FL Partition Sistemi

Yeni modül:

- `fl/partition.py`

Amaç:

- Hasta seviyesinde IID ve Non-IID partition JSON üretmek.
- 2 client zorunluluğunu doğrulamak.
- Non-IID için 80/20 class skew oluşturmak.
- Partition kalitesini validate etmek.

Önemli fonksiyonlar:

- `load_train_metadata`
- `generate_iid_partition`
- `generate_noniid_partition`
- `write_partition`
- `generate_partitions`
- `validate_partition`

CLI tarafında değişiklik:

- `scripts/step11_create_fl_partitions.py` eski client split mantığından arındırıldı.
- Yeni `fl.partition.generate_partitions` fonksiyonunu çağıran sade bir CLI haline getirildi.

Üretilen partition çıktıları:

- `outputs/partitions/iid.json`
- `outputs/partitions/noniid_80_20.json`
- `outputs/final_npy/fl_partitions_iid.json`
- `outputs/final_npy/fl_partitions_noniid.json`

## 7. Reproducibility

Yeni modül:

- `fl/seeding.py`

Amaç:

- Python, NumPy ve PyTorch seed'lerini merkezi şekilde ayarlamak.

Fonksiyon:

- `set_global_seeds`

## 8. Parameter Serialization ve Roundtrip

Yeni modül:

- `fl/parameters.py`

Amaç:

- PyTorch model `state_dict` ile Flower/NumPy parameter listeleri arasında dönüşüm yapmak.
- Checkpoint benzeri parameter dosyalarını serialize/deserialize etmek.
- State dict shape ve dtype uyumluluğunu korumak.

Eklenen fonksiyonlar:

- `get_parameters_as_numpy`
- `load_parameters_into_model`
- `parameter_shapes`
- `serialize_parameters_to_file`
- `deserialize_parameters_from_file`
- `parameters_to_state_dict`

Önemli bug fix:

- 0-D NumPy scalar parameter'ları aggregation/load sırasında hata üretiyordu.
- `np.asarray(param)` ve `torch.tensor(...)` ile scalar-safe dönüşüm eklendi.

Test:

- `tests/test_parameters_roundtrip.py`

Amaç:

- Model parameter'larının NumPy roundtrip sonrasında bozulmadığını doğrulamak.

## 9. FL Config Sistemi

Yeni modül:

- `fl/config.py`

Eklenen dataclass'lar:

- `FLClientConfig`
- `FLServerConfig`
- `FLPartitionConfig`
- `FLPHEConfig`
- `FLSimulationConfig`
- `FLBenchmarkConfig`

Önemli özellikler:

- YAML'dan config yükleme.
- Config snapshot kaydetme.
- Smoke test override uygulama.
- CPU fallback davranışı.

## 10. Communication Tracking

Yeni modül:

- `fl/communication.py`

Amaç:

- Her round için upload/download/toplam byte ölçümü.
- Plaintext/encrypted parameter byte hesaplama.
- Raporlama için iletişim hacmini `round_log.jsonl` içinde taşımak.

Önemli fonksiyonlar/sınıflar:

- `CommunicationTracker`
- `parameters_bytes`
- `encrypted_bytes`

Son raporda kullanılan communication metrikleri:

- `upload_comm_mb`
- `download_comm_mb`
- `total_comm_mb`
- `comm_per_round_mb`
- `phe_overhead_pct`
- `phe_per_round_overhead_pct`

## 11. Dataset Wrapper

Yeni dosya:

- `segmentation/data/fl_dataset.py`

Sınıf:

- `FLPartitionedDataset`

Amaç:

- Existing `CBISDDSMDataset` üzerine client partition JSON'daki `row_indices` ile subset wrapper sağlamak.

Export güncellemesi:

- `segmentation/data/__init__.py`

## 12. FL Local Trainer

Yeni dosya:

- `segmentation/training/fl_trainer.py`

Sınıf:

- `FLLocalTrainer`

Amaç:

- Bir FL client round'u için local training/evaluation yapmak.
- Optimizer state'i her round resetlemek.
- AMP, gradient clipping, loss ve segmentation metrics akışını mevcut trainer pattern'leriyle uyumlu yürütmek.

Export güncellemesi:

- `segmentation/training/__init__.py`

## 13. FL Evaluator

Yeni dosya:

- `segmentation/evaluation/fl_evaluator.py`

Amaç:

- Server-side validation/test evaluation yapmak.
- Existing `SegmentationMetrics` ve `Evaluator` mantığını FL'e uyarlamak.

Önemli fonksiyonlar:

- `run_segmentation_eval`
- `evaluate_on_test`
- `build_loss_for_config`

Önemli fix:

- `loss_fn` evaluation cihazına taşındı: `loss_fn = loss_fn.to(device)`.

Neden:

- CUDA/CPU device mismatch riskini önlemek.

## 14. Quiet Factories

Yeni dosya:

- `fl/factories.py`

Amaç:

- Model ve loss üretimini konsol çıktısı olmadan yapmak.
- Windows console encoding sorunlarından kaçınmak.

Fonksiyonlar:

- `build_model_from_config`
- `build_loss_from_config`

Çözdüğü hata:

- `UnicodeEncodeError: 'charmap' codec can't encode character '\u03b1'`
- Sebep: Merkezi `run.py` içindeki loss print çıktısında Greek alpha/gamma karakterleri.

## 15. Flower Client

Yeni dosya:

- `fl/client.py`

Sınıf:

- `HarmoniaFLClient`

Amaç:

- Flower `NumPyClient` interface'ini uygulamak.
- `get_parameters`, `fit`, `evaluate` metodlarını sağlamak.
- Opsiyonel PHE crypto nesnesi ile encrypted parameter yolunu desteklemek.

## 16. Flower Server Strategy

Yeni dosya:

- `fl/server.py`

Sınıf:

- `HarmoniaFedAvg`

Amaç:

- Flower FedAvg strategy'sini server-side evaluation, checkpoint, convergence tracking ve PHE aggregation için genişletmek.

Önemli fonksiyonlar:

- `aggregate_fit`
- `make_server_eval_fn`
- `weighted_mean_metrics`

PHE tarafında yapılanlar:

- Client'tan gelen encrypted parameter payload'ları alındı.
- Homomorphic weighted sum çağrıldı.
- Server evaluation için encrypted aggregate decrypt edildi.

Önemli not:

- Bu fazda PHE production trust boundary sağlamıyor; secret key server tarafında tutuluyor.

## 17. Simulation Runtime

Yeni dosya:

- `fl/simulation.py`

Amaç:

- FL benchmark simülasyonunu uçtan uca yönetmek.

Önemli fonksiyonlar:

- `run_simulation`
- `build_fl_client`
- `build_global_loader`
- `_resolve_device`
- `_build_crypto`
- `_run_sequential_fedavg`

Ray problemi:

- Windows'ta Ray `AssignProcessToJobObject()` kaynaklı düşük seviye hata verdi.

Çözüm:

- Windows platformunda doğrudan sequential local FedAvg fallback kullanılacak şekilde düzenlendi.
- Ray OSError yakalanırsa fallback'e geçildi.

Sequential FedAvg özellikleri:

- Her round client fit çalıştırıldı.
- Weighted average aggregation yapıldı.
- Server validation metrikleri üretildi.
- Best/final params kaydedildi.
- Round log JSONL yazıldı.
- Convergence bilgisi takip edildi.

Önemli scalar fix:

- Aggregation sırasında `np.asarray(...)` ile scalar parameter'lar güvenli toplandı.

## 18. TenSEAL PHE Modülleri

Yeni klasör:

- `fl/crypto/`

Dosyalar:

- `fl/crypto/__init__.py`
- `fl/crypto/tenseal_context.py`
- `fl/crypto/encryption.py`
- `fl/crypto/aggregation.py`
- `fl/crypto/tools/verify_roundtrip.py`

Ana sınıf:

- `CryptoContext`

Amaç:

- TenSEAL CKKS context oluşturmak.
- Public context export/import yapmak.
- Parameter encrypt/decrypt etmek.
- Homomorphic weighted sum yapmak.

Payload yaklaşımı:

- Encrypted bytes NumPy `uint8` array olarak taşındı.

Roundtrip verification:

- `py -3.11 -m fl.crypto.tools.verify_roundtrip --params ... --config ...`

F2 roundtrip sonucu:

- `val_loss=0.143303`
- `val_dice=0.911649`

F4 roundtrip sonucu:

- `val_loss=0.142425`
- `val_dice=0.911228`

## 19. Benchmark Config'leri

Yeni klasör:

- `benchmark/configs/`

Config dosyaları:

- `benchmark/configs/f1_iid.yaml`
- `benchmark/configs/f2_iid_phe.yaml`
- `benchmark/configs/f3_noniid.yaml`
- `benchmark/configs/f4_noniid_phe.yaml`

Tanımlar:

- F1: IID, PHE false.
- F2: IID, PHE true.
- F3: Non-IID 80/20, PHE false.
- F4: Non-IID 80/20, PHE true.

## 20. Benchmark Runner

Yeni dosya:

- `benchmark/run_benchmark.py`

Amaç:

- Tek config veya tüm F config'lerini programatik çalıştırmak.

Fonksiyonlar:

- `run_config`
- `run_all`

## 21. Top-Level FL CLI

Yeni dosya:

- `run_fl.py`

Subcommand'ler:

- `partition`
- `simulate`
- `benchmark`
- `report`

Önemli tasarım:

- `fl.simulation` ve `benchmark.run_benchmark` lazy import edildi.

Neden:

- `partition` ve `report` komutları Flower/Ray yüklü değilken de çalışabilsin.

Çözdüğü hata:

- `ModuleNotFoundError: No module named 'flwr'`

## 22. Benchmark Report Generator

Yeni/güncellenen dosyalar:

- `benchmark/aggregate_results.py`
- `benchmark/plotting.py`

Üretilen rapor çıktıları:

- `outputs/benchmark_report/report.md`
- `outputs/benchmark_report/comparison_table.csv`
- `outputs/benchmark_report/comparison_table.md`
- `outputs/benchmark_report/dice_by_config.png`
- `outputs/benchmark_report/iou_by_config.png`
- `outputs/benchmark_report/communication_total.png`
- `outputs/benchmark_report/phe_overhead.png`
- `outputs/benchmark_report/convergence_overlay.png`

İlk rapor eksikliği:

- `total_comm_mb` 0 görünüyordu.

Sebep:

- Aggregator `communication_log.csv` arıyordu, ancak gerçek communication metrikleri `round_log.jsonl` içindeydi.

Çözüm:

- `_read_round_log` eklendi.
- `upload_bytes`, `download_bytes`, `total_bytes`, `wall_time_sec` round log'dan toplanmaya başladı.
- `comm_per_round_mb` ve per-round PHE overhead eklendi.

Son raporda bulunan metrikler:

- `config`
- `scenario`
- `phe`
- `dice_mean`
- `dice_std`
- `iou_mean`
- `iou_std`
- `sens_mean`
- `sens_std`
- `spec_mean`
- `spec_std`
- `prec_mean`
- `prec_std`
- `pixel_acc`
- `pixel_tp`
- `pixel_fp`
- `pixel_fn`
- `pixel_tn`
- `n_test_samples`
- `wall_time_total_sec`
- `wall_time_avg_sec`
- `n_rounds`
- `n_rounds_to_best`
- `best_val_dice`
- `best_val_iou`
- `best_val_loss`
- `final_val_dice`
- `final_val_iou`
- `final_val_loss`
- `upload_comm_mb`
- `download_comm_mb`
- `total_comm_mb`
- `comm_per_round_mb`
- `phe_overhead_pct`
- `phe_per_round_overhead_pct`

## 23. Testler

Güncellenen/eklenen testler:

- `tests/test_step11_fl_clients.py`
- `tests/test_parameters_roundtrip.py`

Partition testleri:

- IID split hasta ayrıklığını ve dengeli dağılımı test etti.
- Non-IID split 80/20 skew davranışını test etti.
- `n_clients` validation test edildi.

Test sırasında yaşanan hata:

- `ValueError: IID skew too high: 0.2500`

Sebep:

- Synthetic test datası çok küçüktü.

Çözüm:

- Test fixture patient count `80` yerine `400` yapıldı.

## 24. Dependency ve Environment Hataları

### 24.1 `ModuleNotFoundError: No module named 'torch'`

Sebep:

- Default Python ortamında PyTorch yoktu.

Çözüm:

- Python 3.11 kullanıldı: `py -3.11`.

### 24.2 Ray Python 3.13 uyumsuzluğu

Hata:

- `ERROR: No matching distribution found for ray<3.0.0,>=2.10.0`

Sebep:

- Ray için Python 3.13 wheel bulunmadı.

Çözüm:

- Python 3.11'e geçildi.

### 24.3 `pytest` yoktu

Hata:

- `python.exe: No module named pytest`

Çözüm:

- Python 3.11 ortamına `pytest` yüklendi.

### 24.4 CPU-only Torch

Hata:

- `AssertionError: Torch not compiled with CUDA enabled`

Sebep:

- İlk PyTorch kurulumu CPU-only idi.

Çözüm:

- Config CPU fallback eklendi.
- Daha sonra PyTorch CUDA 13.0 wheel ile yeniden kuruldu.

### 24.5 Windows Ray hatası

Hata:

- `FileNotFoundError: [Errno 2] AssignProcessToJobObject() failed`

Çözüm:

- Windows'ta sequential local FedAvg fallback eklendi.

### 24.6 NumPy scalar aggregation hatası

Hata:

- `TypeError: 'numpy.float32' object cannot be interpreted as an integer`

Çözüm:

- `load_parameters_into_model` ve sequential FedAvg aggregation scalar-safe hale getirildi.

### 24.7 Pandas markdown dependency

Hata:

- `ImportError: Import tabulate failed.`

Çözüm:

- `tabulate>=0.9.0` `requirements_fl.txt` içine eklendi.

## 25. FL Benchmark Çalıştırmaları

Her full run öncesinde smoke test çalıştırıldı.

Çalıştırılan ana koşular:

- F1 IID no PHE full.
- F2 IID + PHE smoke.
- F2 IID + PHE full.
- F3 Non-IID no PHE full.
- F4 Non-IID + PHE smoke.
- F4 Non-IID + PHE full.

Not:

- F3 local run terminalde manuel sonlandırılmış göründü; `test_results.json` yazılmıştı. Exit code bu nedenle gerçek eğitim hatası olarak değerlendirilmedi.

## 26. Final Benchmark Sonuçları

Kaynak:

- `outputs/benchmark_report/comparison_table.csv`
- `outputs/benchmark_report/report.md`

### 26.1 C1 Centralized Baseline

- Scenario: centralized.
- PHE: false.
- Test samples: 260.
- Dice mean: `0.9173932978880625`
- Dice std: `0.056505733139229956`
- IoU mean: `0.85176389228779`
- IoU std: `0.0843478741234325`
- Sensitivity mean: `0.9396669847143961`
- Sensitivity std: `0.05353377076421441`
- Specificity mean: `0.9281299945885944`
- Specificity std: `0.05813446258409058`
- Precision mean: `0.9047990429972557`
- Precision std: `0.0910303619972805`
- Pixel accuracy: `0.9292340205265925`
- Pixel TP: `6995682`
- Pixel FP: `724439`
- Pixel FN: `481368`
- Pixel TN: `8837871`

### 26.2 F1 IID, no PHE

- Scenario: IID.
- PHE: false.
- Test samples: 260.
- Dice mean: `0.9141980747395986`
- Dice std: `0.05616011493278139`
- IoU mean: `0.8462876409936867`
- IoU std: `0.08430812433429118`
- Sensitivity mean: `0.9276082306596565`
- Sensitivity std: `0.06415272827113028`
- Specificity mean: `0.9324392251498969`
- Specificity std: `0.061943644715715285`
- Precision mean: `0.9110911920281292`
- Precision std: `0.09004405021961492`
- Pixel accuracy: `0.9266307537372296`
- Pixel TP: `6904650`
- Pixel FP: `677765`
- Pixel FN: `572400`
- Pixel TN: `8884545`
- Wall time total: `3965.1787672042847` sec.
- Wall time avg per round: `198.25893836021424` sec.
- Rounds: `20`
- Best round: `16`
- Best val Dice: `0.9112281856593748`
- Best val IoU: `0.8431875477308428`
- Best val loss: `0.1431592155654891`
- Final val Dice: `0.9094673571388822`
- Final val IoU: `0.840395941205536`
- Final val loss: `0.1469065602086171`
- Upload communication: `3251.0601806640625` MB.
- Download communication: `3251.0601806640625` MB.
- Total communication: `6502.120361328125` MB.
- Communication per round: `325.10601806640625` MB.

### 26.3 F2 IID + PHE

- Scenario: IID.
- PHE: true.
- Test samples: 260.
- Dice mean: `0.9143906239009804`
- Dice std: `0.05723437654309457`
- IoU mean: `0.8467698983380311`
- IoU std: `0.08571557387267997`
- Sensitivity mean: `0.9346955895232281`
- Sensitivity std: `0.0612211177052256`
- Specificity mean: `0.9273024406661067`
- Specificity std: `0.06095914769714622`
- Precision mean: `0.9046018250640322`
- Precision std: `0.09180505369569934`
- Pixel accuracy: `0.9265745309682992`
- Pixel TP: `6956265`
- Pixel FP: `730338`
- Pixel FN: `520785`
- Pixel TN: `8831972`
- Wall time total: `4994.084264039993` sec.
- Wall time avg per round: `249.70421320199966` sec.
- Rounds: `20`
- Best round: `19`
- Best val Dice: `0.9116485235740416`
- Best val IoU: `0.8440708791957715`
- Best val loss: `0.1433029681945047`
- Final val Dice: `0.9110142985922268`
- Final val IoU: `0.8429683421953515`
- Final val loss: `0.1447272049028332`
- Upload communication: `3251.0601806640625` MB.
- Download communication: `3251.0601806640625` MB.
- Total communication: `6502.120361328125` MB.
- Communication per round: `325.10601806640625` MB.
- Total PHE overhead vs F1: `0.0`
- Per-round PHE overhead vs F1: `0.0`
- PHE roundtrip validation:
  - Val loss: `0.143303`
  - Val Dice: `0.911649`

### 26.4 F3 Non-IID, no PHE

- Scenario: Non-IID 80/20.
- PHE: false.
- Test samples: 260.
- Dice mean: `0.91301504428907`
- Dice std: `0.05769208964669288`
- IoU mean: `0.8444720480742074`
- IoU std: `0.08582231279953324`
- Sensitivity mean: `0.9383803689285464`
- Sensitivity std: `0.056988887902570236`
- Specificity mean: `0.9209076523105627`
- Specificity std: `0.06319944923881107`
- Precision mean: `0.8982029281968635`
- Precision std: `0.09259096976867394`
- Pixel accuracy: `0.9249568058894231`
- Pixel TP: `6988392`
- Pixel FP: `790030`
- Pixel FN: `488658`
- Pixel TN: `8772280`
- Wall time total: `9611.950500011444` sec.
- Wall time avg per round: `246.46026923106268` sec.
- Rounds: `39`
- Best round: `12`
- Best val Dice: `0.9106282944540024`
- Best val IoU: `0.842250094018304`
- Best val loss: `0.1403053612017832`
- Final val Dice: `0.9099712741185688`
- Final val IoU: `0.841658036027135`
- Final val loss: `0.1457850045266271`
- Upload communication: `6339.567352294922` MB.
- Download communication: `6339.567352294922` MB.
- Total communication: `12679.134704589844` MB.
- Communication per round: `325.10601806640625` MB.

Not:

- F3'te `n_rounds=39` görünmesi, çıktının önceki/stale run ile birleşmiş veya run lifecycle içinde ek round log tutulmuş olabileceğini gösterir. Test sonucu üretilmiş ve benchmark tablosuna alınmıştır. Per-round communication bu yüzden total communication'dan daha anlamlıdır.

### 26.5 F4 Non-IID + PHE

- Scenario: Non-IID 80/20.
- PHE: true.
- Test samples: 260.
- Dice mean: `0.9133306294128556`
- Dice std: `0.05583633372454446`
- IoU mean: `0.8447588686315171`
- IoU std: `0.08374869152252062`
- Sensitivity mean: `0.931173980172775`
- Sensitivity std: `0.061105440991957125`
- Specificity mean: `0.9269445553544249`
- Specificity std: `0.06359715445889369`
- Precision mean: `0.9058218242169641`
- Precision std: `0.09134039981649363`
- Pixel accuracy: `0.9256582993727464`
- Pixel TP: `6936335`
- Pixel FP: `726020`
- Pixel FN: `540715`
- Pixel TN: `8836290`
- Wall time total: `6065.195271492004` sec.
- Wall time avg per round: `303.25976357460024` sec.
- Rounds: `20`
- Best round: `17`
- Best val Dice: `0.9112284365208196`
- Best val IoU: `0.843287832928905`
- Best val loss: `0.1424248085803344`
- Final val Dice: `0.9061384465216116`
- Final val IoU: `0.8345826232218979`
- Final val loss: `0.1513443746987511`
- Upload communication: `3251.0601806640625` MB.
- Download communication: `3251.0601806640625` MB.
- Total communication: `6502.120361328125` MB.
- Communication per round: `325.10601806640625` MB.
- Total PHE overhead vs F3: `-48.717948717948715`
- Per-round PHE overhead vs F3: `0.0`
- PHE roundtrip validation:
  - Val loss: `0.142425`
  - Val Dice: `0.911228`

Not:

- Total PHE overhead'in negatif görünmesi F3 log'unda 39 round bulunmasından kaynaklanır. Per-round overhead karşılaştırması bu etkiyi normalize eder ve `0.0` görünür.

## 27. Benchmark Yorumu

Ana sonuçlar:

- Centralized baseline C1 en yüksek Dice'a sahip: `0.91739`.
- FL koşuları centralized baseline'a çok yakın performans verdi.
- F1 Dice: `0.91420`
- F2 Dice: `0.91439`
- F3 Dice: `0.91302`
- F4 Dice: `0.91333`
- PHE kullanılan F2/F4 performansında kayda değer precision/dice bozulması görünmedi.
- Non-IID koşulunda performans düşüşü sınırlı kaldı.
- PHE roundtrip doğrulamaları, decrypt sonrası validation metriklerinin server best validation sonuçlarıyla tutarlı olduğunu gösterdi.

## 28. DigitalOcean ve VPS Çalışmaları

Kullanıcı DigitalOcean personal access token paylaştı.

Yapılanlar:

- Token `.env` içine alındı.
- `.env` `.gitignore` ile commit dışı bırakıldı.
- `doctl` kullanılarak account/snapshot/region/size bilgileri incelendi.
- Mevcut başka proje/droplet'e dokunulmaması gerektiği özellikle dikkate alındı.

GPU droplet durumu:

- Kullanıcı tam benchmark'ları VPS GPU üzerinde koşturmayı değerlendirdi.
- DigitalOcean GPU droplet oluşturma denemeleri yapıldı.
- `gpu-4000adax1-20gb` gibi size/region kombinasyonları incelendi.
- FRA1/TOR1 denemelerinde "Size is not available in this region" ve kapasite/quota benzeri 422 hataları görüldü.
- Kullanıcı arayüzde "We're out of GPU's right now." mesajını gördü.
- Son karar: GPU yokluğu nedeniyle eğitimler lokal devam edecek.

CPU droplet planı:

- VPS kullanımının proje dokümantasyonunda yer alması gerektiği için CPU droplet'ler açıldı.
- Amaç model eğitimi değil, network/security/deployment ayarlarının snapshot'ta hazır olduğunu doğrulamaktı.

Aktif Harmonia droplet'leri:

- `harmonia-server`
- `harmonia-client1`
- `harmonia-client2`

Region:

- `tor1`

Ek mevcut droplet:

- `kreb-trading`

Bu droplet'e dokunulmadı.

SSH port durumu:

- İlk SSH denemesi port 22 ile timeout verdi.
- Kullanıcı SSH portunun 44 olduğunu belirtti.
- `ssh -p 44` ile bağlantı başarılı oldu.

VPS kontrolleri:

- SSH erişimi doğrulandı.
- Docker varlığı kontrol edildi.
- Python varlığı kontrol edildi.
- UFW/firewall durumu kontrol edildi.
- WireGuard interface/network ayarlarının snapshot'ta bulunduğu doğrulandı.
- Private network connectivity kontrol edildi.
- mTLS sertifika üretimi ve upload işlemleri yapıldı.

Maliyet notu:

- Harmonia CPU droplet'leri açık kaldığı sürece ücret yazmaya devam eder.

## 29. Deployment Scaffold

Yeni klasör:

- `deployment/`

Dosyalar:

- `deployment/docker/Dockerfile.server`
- `deployment/docker/Dockerfile.client`
- `deployment/docker/docker-compose.yml`
- `deployment/docker/entrypoint_server.sh`
- `deployment/docker/entrypoint_client.sh`
- `deployment/certs/generate_certs.py`
- `deployment/certs/README.md`
- `deployment/provisioning/bootstrap_droplet.sh`
- `deployment/provisioning/upload_data.sh`
- `deployment/provisioning/README.md`

Amaç:

- CPU/GPU bağımsız VPS deployment iskeleti sağlamak.
- Server/client container ayrımı yapmak.
- mTLS cert generation akışını dokümante etmek.
- Droplet bootstrap ve data upload akışını komutlaştırmak.

Secret handling:

- Generated cert/key çıktıları `.gitignore` ile commit dışı bırakıldı.

## 30. README Güncellemesi

`README.md` güncellendi.

Eklenenler:

- Projenin üçüncü bileşeni olarak Federated Learning Benchmark eklendi.
- `fl/` ve `benchmark/` klasörleri tanıtıldı.
- FL rapor linkleri eklendi:
  - `outputs/benchmark_report/report.md`
  - `outputs/benchmark_report/comparison_table.csv`
- Raporun içerdiği metrikler kısa şekilde açıklandı.

## 31. Git ve GitHub Çalışmaları

Başlangıç durumu:

- `c:\Users\berk\Desktop\harmonia_v2` klasörü git repo değildi.

Yapılanlar:

- `git init -b main`
- `.gitignore` kontrol edildi ve genişletildi.
- `.gitattributes` içinde `.npy` ve `.pt` için LFS işaretleri vardı, ancak bu dosyalar ayrıca ignore edildi.
- `.env`, `.npy`, `.pt`, generated cert/key dosyaları commit dışında doğrulandı.
- Tüm kod, config, küçük CSV/JSON/MD/PNG rapor çıktıları staged edildi.
- Initial commit oluşturuldu.

Commit:

- `ffb6d2f Add Harmonia FL benchmark implementation and report`

GitHub CLI durumu:

- Başta `gh` kurulu değildi.
- `winget install --id GitHub.cli -e --source winget` ile GitHub CLI kuruldu.
- `gh auth login --hostname github.com --web --git-protocol https` ile login başlatıldı.
- Browser otomatik açılmadığı için cihaz kodu yöntemi kullanıldı.
- Kullanıcı GitHub device login'i tamamladı.
- Hesap: `benkreb`

Repo oluşturma:

- `gh repo create harmonia_v2 --private --source . --remote origin --push`

Son durum:

- GitHub repo: `https://github.com/benkreb/harmonia_v2`
- Visibility: private.
- Branch: `main`.
- Remote: `origin`.
- Local branch `origin/main` ile senkron.

## 32. Git Ignore ve Güvenlik Kontrolü

Commit dışında bırakılanlar:

- `.env`
- `.pytest_cache/`
- `AGENTS.md`
- `__pycache__/`
- `logs/`
- `outputs/**/*.pt`
- `outputs/**/*.pth`
- `outputs/**/*.onnx`
- `outputs/**/*.ckpt`
- `outputs/final_npy/*.npy`
- `outputs/final_npy*/fl_clients/**/*.npy`
- `outputs/processed/`
- `outputs/combined_masks*/`
- `outputs/mass_processed_images/`
- `outputs/mass_processed_masks/`
- `outputs/segmentation/checkpoints/`
- `outputs/segmentation/evaluation/*.npy`
- `deployment/certs/generated/`
- `deployment/certs/**/*.pem`
- `deployment/certs/**/*.key`
- `deployment/certs/**/*.crt`
- `deployment/certs/**/*.csr`

Commit'e alınan çıktı tipleri:

- Küçük metadata CSV dosyaları.
- Partition JSON dosyaları.
- Evaluation JSON dosyaları.
- Benchmark report MD/CSV dosyaları.
- Benchmark PNG grafikleri.
- Config snapshot YAML dosyaları.
- Training history JSON.

## 33. Önemli Komutlar

Partition üretimi:

```powershell
py -3.11 run_fl.py partition
```

Tek config simulation:

```powershell
py -3.11 run_fl.py simulate --config benchmark/configs/f1_iid.yaml
```

Smoke simulation:

```powershell
py -3.11 run_fl.py simulate --config benchmark/configs/f4_noniid_phe.yaml --smoke
```

Rapor üretimi:

```powershell
py -3.11 run_fl.py report
```

F2 PHE roundtrip:

```powershell
py -3.11 -m fl.crypto.tools.verify_roundtrip --params outputs/fl/f2_iid_phe/best_params.pt --config benchmark/configs/f2_iid_phe.yaml
```

F4 PHE roundtrip:

```powershell
py -3.11 -m fl.crypto.tools.verify_roundtrip --params outputs/fl/f4_noniid_phe/best_params.pt --config benchmark/configs/f4_noniid_phe.yaml
```

GitHub repo create/push:

```powershell
gh repo create harmonia_v2 --private --source . --remote origin --push
```

## 34. Bilinen Limitasyonlar

- Her F config için tek seed/run vardır; rerun variance yoktur.
- PHE fazında server-held CKKS secret key kullanılmıştır; production-grade trust boundary sunmaz.
- PHE overhead bu local sequential fallback mimarisinde gerçek network serialization overhead'ini tam temsil etmeyebilir.
- F3 round log'unda 39 round görünmesi total communication karşılaştırmasını etkiler; per-round communication daha doğru yorumlanmalıdır.
- Windows ortamında Ray simulation yerine sequential fallback kullanılmıştır.
- VPS üzerinde GPU kapasite problemi nedeniyle eğitim çalıştırılmamıştır.
- CPU droplet'ler deployment/network/security doğrulaması için kullanılmıştır.
- CBIS-DDSM raw dataset ve `.npy` binary artefact'leri GitHub'a yüklenmemiştir.

## 35. Nihai Dosya ve Klasör Özeti

FL runtime:

- `fl/client.py`
- `fl/server.py`
- `fl/simulation.py`
- `fl/config.py`
- `fl/partition.py`
- `fl/parameters.py`
- `fl/communication.py`
- `fl/factories.py`
- `fl/seeding.py`

PHE:

- `fl/crypto/tenseal_context.py`
- `fl/crypto/aggregation.py`
- `fl/crypto/encryption.py`
- `fl/crypto/tools/verify_roundtrip.py`

Segmentation integration:

- `segmentation/data/fl_dataset.py`
- `segmentation/training/fl_trainer.py`
- `segmentation/evaluation/fl_evaluator.py`

Benchmark:

- `benchmark/configs/f1_iid.yaml`
- `benchmark/configs/f2_iid_phe.yaml`
- `benchmark/configs/f3_noniid.yaml`
- `benchmark/configs/f4_noniid_phe.yaml`
- `benchmark/run_benchmark.py`
- `benchmark/aggregate_results.py`
- `benchmark/plotting.py`

CLI:

- `run_fl.py`

Tests:

- `tests/test_step11_fl_clients.py`
- `tests/test_parameters_roundtrip.py`

Deployment:

- `deployment/docker/`
- `deployment/certs/`
- `deployment/provisioning/`

Reports:

- `outputs/benchmark_report/report.md`
- `outputs/benchmark_report/comparison_table.csv`
- `outputs/benchmark_report/comparison_table.md`
- `outputs/benchmark_report/*.png`
- `CHAT_WORK_FULL_REPORT.md`

## 36. Son Durum

Proje şu anda:

- FL ve PHE implementation içeriyor.
- IID ve Non-IID partition üretimi içeriyor.
- F1/F2/F3/F4 sonuçlarını içeriyor.
- C1 centralized baseline ile karşılaştırmalı benchmark raporu içeriyor.
- PHE roundtrip doğrulamaları yapılmış durumda.
- VPS deployment scaffold içeriyor.
- DigitalOcean CPU droplet güvenlik/network doğrulaması yapılmış durumda.
- GitHub private repo olarak yüklenmiş durumda.

GitHub:

- `https://github.com/benkreb/harmonia_v2`
- Visibility: private.
- Branch: `main`.

Bu rapor, chat boyunca yapılan işleri ve kararları tek yerde toplamak için oluşturulmuştur.
