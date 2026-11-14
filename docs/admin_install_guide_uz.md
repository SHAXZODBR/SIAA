> **Taqrizchi uchun eslatma:** ushbu hujjatdagi klinik atamalarni klinikaga topshirishdan oldin amaliyotchi rentgenolog shifokor tekshirib chiqishi shart.

# SIAA — Sentinel Medical AI · O‘rnatish va ma’muriy boshqaruv qo‘llanmasi (kasalxona IT xizmati)

Versiya 1.1.0 · Hujjat tahriri 2026-11

Ushbu qo‘llanma klinikaning IT administratori uchun mo‘ljallangan. Rentgenolog shifokorlar uchun qo‘llanma — `user_manual_uz.md`.

---

## 1. Arxitektura bir abzatsda

Hamma narsa **bitta klinika kompyuterida** ishlaydi (rentgenolog shifokorning ish stantsiyasi yoki uning yonidagi kichik server):

- **Ish stoli ilovasi** (Electron + React) — foydalanuvchi interfeysi. Yig‘ilgan distributivda u AI serverini o‘zi ishga tushiradi.
- **Python AI serveri** (FastAPI, PyTorch), **127.0.0.1:8000** manzilini tinglaydi. DICOM ni o‘qiydi, bosh miya detektorlari panelini ishga tushiradi, hisobot qoralamalarini yaratadi, tekshiruvlar, imzolangan hisobotlar va audit jurnalini mahalliy SQLite bazasida saqlaydi.
- **Modellar papkasi** — o‘rnatish bilan birga yetkaziladigan, faqat o‘qish uchun og‘irlik (weight) fayllari.
- **Ollama** va Gemma 3 til modeli **localhost:11434** da — faqat hisobot matni, tarjima va *AI dan so‘rash* yorlig‘i uchun ishlatiladi. Ixtiyoriy: usiz server qat’iy hisobot shablonlariga o‘tadi.

Ishlash davomida internet ulanishi talab qilinmaydi va ishlatilmaydi. Hech narsa hech qayerga yuklanmaydi.

```
tomograf eksporti / USB / umumiy papka → Ish stoli ilovasi → 127.0.0.1:8000 AI serveri → SQLite + jurnallar (ma’lumotlar papkasi)
                                                                        └→ localhost:11434 Ollama (faqat hisobot matni)
```

---

## 2. Kompyuterga minimal talablar

| Element | Minimal | Tavsiya etiladi |
|---|---|---|
| OT | Windows 10/11 64-bit (NSIS o‘rnatuvchi, x64) yoki macOS 12+ (DMG, arm64/x64) | Windows 11 Pro (BitLocker mavjud) |
| Protsessor | 4 yadroli x86-64 / Apple silicon. **Faqat protsessorda (CPU) inferens ishlaydi**; GPU talab qilinmaydi. | 8 yadro |
| Operativ xotira | 8 GB (AI serveri + ish stoli ilovasi) | Ollama shu kompyuterda ishlasa — **16 GB** (Gemma 3 4B ning o‘ziga ~8 GB kerak) |
| Disk | SSD da 20 GB bo‘sh joy: ilova + Python backend ~3 GB, modellar ~0,5–2 GB, Ollama modeli ~3,3 GB, shuningdek o‘sib boruvchi ma’lumotlar papkasi | 50 GB bo‘sh SSD |
| GPU | Ixtiyoriy. Mavjud bo‘lsa CUDA (NVIDIA) yoki Apple MPS avtomatik ishlatiladi (`/health` dagi `device`). | — |
| Tarmoq | Talab qilinmaydi. 8000 va 11434 portlari faqat localhost ga bog‘langan. | — |
| Displey | 1600×1000 yoki kattaroq (ilova to‘liq ekranda ochiladi; minimal oyna 1200×800) | — |

CPU da bir bosh miya tekshiruvi uchun inferens vaqti — bir necha soniyadan o‘nlab soniyagacha; u har bir natija bilan ko‘rsatiladi.

---

## 3. Nima o‘rnatiladi

| Komponent | Joylashuvi (Windows) | Izohlar |
|---|---|---|
| Ish stoli ilovasi | Program Files (yoki foydalanuvchi papkasi) — `Sentinel Medical AI` | «Пуск» menyusi va ish stoli yorliqlari. O‘rnatuvchi ikki tilli tibbiy ogohlantirishni ko‘rsatadi; uni qabul qilish shart. |
| Python backend | `<o‘rnatish>\resources\backend\sentinel-backend\` — bitta dasturga «muzlatilgan» AI serveri (`sentinel-backend.exe`, o‘zining Python muhiti bilan; manba koddan o‘rnatishda uning o‘rnida `run_server.py`, `src\` va `venv\`) | Ilova ishga tushganda ishga tushiriladi. Uning stdout/stderr chiqishi `<ma’lumotlar papkasi>\logs\backend.log` ga yoziladi. |
| Modellar | `<o‘rnatish>\resources\backend\models\` (`SENTINEL_MODELS_DIR` bilan o‘zgartirish mumkin) | `brain_triage_finetuned\` (tasdiqlangan triaj, ~330 MB, `MANIFEST.json` bilan — o‘rnatuvchida faqat *to‘liq* variantda, quyida qarang), `hf\` (ommaviy o‘sma klassifikatori to‘plami), ixtiyoriy boshqalar. |
| Ollama + Gemma 3 | Alohida o‘rnatuvchi (ollama.com); model `ollama pull gemma3:4b` bilan yuklanadi | Kompyuter internetdan uzilishidan **oldin** o‘rnatilishi shart. |
| Ma’lumotlar papkasi | 6-bandga qarang | Birinchi ishga tushirishda yaratiladi. |

macOS da mos yo‘llar `Sentinel Medical AI.app/Contents/Resources/backend` ichida.

**O‘rnatuvchining ikki varianti.** Yetkazib beruvchi bitta ilovani ikki variantda yig‘adi; qaysi biri olinganini yetkazib beruvchining ilova xati ko‘rsatadi (o‘rnatuvchi fayli ikkala variantda bir xil nomlanadi):

| Variant | Yig‘ish natijasi | Tasdiqlangan triaj modeli | Nima uchun |
|---|---|---|---|
| **To‘liq** | `sentinel-windows-x64-full` — endi yuklab olinadigan ish jarayoni artefakti emas: faqat `v*` tegi bo‘yicha yig‘iladi va shu tegning GitHub Release *qoralamasiga* biriktiriladi (13.1-band) | O‘rnatuvchi ichida. Birinchi ishga tushirishda shu kompyuter uchun avtomatik shifrlanadi (8.3-band). | Yetkazib beruvchi shaxsan bajaradigan o‘rnatishlar, namoyishlar. |
| **Klinika** | `sentinel-windows-x64-NO-VALIDATED-MODEL` (ish jarayoni artefakti, 1 kun saqlanadi) | O‘rnatuvchida **yo‘q**. USB orqali alohida, *faqat shu kompyuter uchun* shifrlangan paket sifatida, juft `license.dat` bilan birga yetkaziladi (5.3, 5.4-bandlar). | Har qanday klinikaga joriy etish: o‘rnatuvchining o‘zida yetkazib beruvchining xususiy modeli yo‘q va uni erkin nusxalash mumkin. |

Paket import qilinmaguncha klinika variantidagi o‘rnatish `/health` → `"status": "degraded"` deb xabar beradi va *tasdiqlangan model yuklanmadi* bannerini ko‘rsatadi — bu kutilgan holat va 5.4-banddan keyin yo‘qoladi.

---

## 4. To‘liq avtonom (oflayn) ishlash

- Server **standart holatda oflayn rejimda** ishlaydi (dasturchi rejimidan tashqari har qanday ishga tushirishda `SENTINEL_OFFLINE=1`). Hugging Face va transformers import paytida oflayn rejimga o‘tkaziladi; **ish vaqtida hech qanday model hech qachon yuklab olinmaydi**.
- Shuning uchun har bir model birinchi foydalanishdan oldin modellar papkasida bo‘lishi shart. Model yo‘qligi serverni to‘xtatmaydi: `/health` uni yuklanmagan deb, *«model missing — run scripts/download_all_models.py --only <key> on a machine with internet and copy models/ here»* sababi bilan ko‘rsatadi.
- Bulutli til modellari o‘chirilgan (`SENTINEL_ALLOW_CLOUD_LLM` berilmagan). Hisobot matni localhost dagi Ollama dan yoki shablonlardan keladi.
- Agar modellar papkasini internetga **ulangan** kompyuterda tayyorlasangiz, u yerda `python scripts/download_all_models.py` ni bajaring va hosil bo‘lgan `models\` papkasini klinika kompyuteriga nusxalang.

---

## 5. Birinchi ishga tushirish

### 5.1 Administrator hisob qaydnomasi

Eng birinchi ishga tushirishda `admin` hisob qaydnomasi (roli *admin*) yaratiladi:

- Agar server birinchi marta ishga tushganda **`SENTINEL_ADMIN_PASSWORD`** muhit o‘zgaruvchisi berilgan bo‘lsa, uning qiymati administrator paroli bo‘ladi.
- Aks holda **tasodifiy bir martalik parol** yaratiladi va **bir marta** chop etiladi: server konsoliga va, WARNING darajasidagi qatorlar sifatida, `<ma’lumotlar papkasi>\logs\sentinel_YYYY-MM-DD.log` jurnal fayliga (yig‘ilgan distributivda — `<ma’lumotlar papkasi>\logs\backend.log` ga ham). *FIRST-RUN ADMIN ACCOUNT CREATED (one-time password)* bannerini qidiring.
- Har ikkala holda ham hisob qaydnomasi **parolni o‘zgartirish shart** deb belgilanadi: birinchi kirishda parolni o‘zgartirish shakli ochiladi va parol almashtirilmaguncha boshqa hech narsa ishlamaydi (ilovada kamida 8 ta belgi).

Bir martalik parolni boshqalar o‘qiy oladigan faylda hech qachon qoldirmang. Jurnal qatori — u saqlanadigan yagona joy.

### 5.2 Dastlabki sozlash ustasi

Birinchi ishga tushirishda ish stoli ilovasi **Dastlabki sozlash** ustasini ochadi (*Sozlamalar → Klinika → Sozlashni qayta boshlash* orqali qayta ochish mumkin):

1. **Til** — interfeys tili va hisobotlarning standart tili (RU / UZ / EN).
2. **Klinika** — klinika nomi (majburiy; har bir PDF sarlavhasida chop etiladi), manzil, telefon.
3. **AI serveri** — server URL manzili; server shu kompyuterda ishlasa, `http://127.0.0.1:8000` ni qoldiring. Usta serverning joriy holatini ko‘rsatadi.
4. **Litsenziya** — kompyuter barmoq izi va `license.dat` importi (5.3-band).
5. **Qo‘llab-quvvatlash** — shifokorlarga har bir xato bannerida va kirish ekranida ko‘rsatiladigan e-mail/telefon. Bu yerga o‘z yordam xizmatingizni yozing.

[Skrinshot: dastlabki sozlash ustasi, 3-qadam «AI serveri», yashil holat bilan]

### 5.3 Litsenziyani faollashtirish

Birinchi ishga tushirishda server hech narsani rad etmaydi: **amaldagi litsenziyasiz u demo rejimida ishlaydi — kuniga 10 ta tahlil**, undan keyin tahlil so‘rovlari keyingi kungacha rad etiladi (HTTP 402). Imzolash va ko‘rish ishlashda davom etadi.

Litsenziya — yetkazib beruvchi tomonidan RSA bilan imzolangan va **ushbu kompyuterning barmoq iziga bog‘langan** **`license.dat`** fayli (xost nomi, OT, protsessor modeli, asosiy MAC manzil, OT machine ID).

1. Barmoq izini oling. Yoki *Sozlamalar → Litsenziya* da o‘qing (ilova AI serveridan so‘raydi), yoki kompyuterda `<o‘rnatish>\resources\backend\sentinel-backend\sentinel-backend.exe --print-fingerprint` ni (o‘rnatuvchidan yig‘ilma) yoki manba koddan o‘rnatishda quyidagini bajaring:
   ```
   cd <o‘rnatish>\resources\backend
   venv\Scripts\python.exe -m src.utils.license fingerprint
   ```
2. Barmoq izi satrini (64 ta hex belgi) va klinika nomini yetkazib beruvchiga yuboring.
3. Qaytarilgan `license.dat` ni **ma’lumotlar papkasiga** (6-band) saqlang — *Sozlamalar → Litsenziya → license.dat ni import qilish* orqali yoki faylni u yerga qo‘lda nusxalab.
4. Ilovani qayta ishga tushiring (bu serverni ham qayta ishga tushiradi). `/health` → `license.mode` qiymati `licensed` ekanini yoki *Sozlamalar → Litsenziya* da *Litsenziyalangan*, mijoz nomi va amal qilish muddati ko‘rsatilganini tekshiring.
5. Faylni qo‘lda tekshirish: `venv\Scripts\python.exe -m src.utils.license verify --license-file <yo‘l>`.

Kompyuter barmoq izi o‘zgarsa (yangi tarmoq kartasi, xost nomi o‘zgartirilgan, klonlangan VM) yoki amal qilish muddati tugasa litsenziya kuchini yo‘qotadi — sababi *Sozlamalar → Litsenziya* da va jurnalda ko‘rsatiladi. Qayta berish uchun yetkazib beruvchiga murojaat qiling; bu orada dasturiy ta’minot demo rejimiga o‘tadi, to‘xtab qolmaydi.

**Klinika varianti: litsenziya modelni ham ochadi.** Klinika varianti uchun yetkazib beruvchi litsenziyani `python -m src.utils.license issue … --with-model-key` buyrug‘i bilan chiqaradi (buni uning `packaging/provision_clinic.py` vositasi bajaradi). Shunda imzolangan litsenziya ushbu kompyuterning model kaliti hosil qilinadigan sirni o‘z ichiga oladi, 5.4-banddagi model paketi esa aynan shu kalit bilan shifrlangan — litsenziya va paket bitta yetkazib berishdagi **juftlik**dir. Oqibatlari:

- Avval litsenziyani import qiling, ilovani qayta ishga tushiring, so‘ngra paketni import qiling (5.4-band).
- Litsenziya kuchini yo‘qotsa (muddati tugadi, apparat yoki xost nomi o‘zgardi), tasdiqlangan modelni endi deshifrlab bo‘lmaydi: `/health` `degraded` holatiga o‘tadi va triaj detektori to‘xtaydi. **Litsenziyani muddati tugashidan kamida 30 kun oldin uzaytiring**; apparat almashtirilgandan keyin yangi barmoq izini yuboring va yetkazib beruvchi qaytargan yangi litsenziya va paketni import qiling.
- Klinika variantidagi o‘rnatishning litsenziyasini hech qachon boshqa yetkazib berishdagi litsenziya bilan almashtirmang: boshqa kalitli litsenziya juft paket import qilinmaguncha o‘rnatilgan paketni o‘qib bo‘lmaydigan qiladi.
- To‘liq variantdagi o‘rnatish odatda litsenziyani `--with-model-key` **siz** oladi (uning modeli birinchi ishga tushirishda o‘rnatish kaliti bilan shifrlanadi, 8.3-band). Agar baribir shunday litsenziya olsa, server keyingi ishga tushirishda modelni litsenziya kaliti bilan qayta shifrlaydi va shundan keyin model yuqorida tasvirlanganidek litsenziyaga bog‘liq bo‘ladi.

### 5.4 Shifrlangan model paketini o‘rnatish (klinika varianti)

Har bir kompyuter uchun yetkazib beruvchi `<klinika>_<barmoq izining dastlabki 8 belgisi>` nomli papkani topshiradi; unda `license.dat`, `brain_triage_finetuned_<identity12>_<fp8>.zip` va `README_admin_ru.txt` / `README_admin_en.txt` yo‘riqnomalari bo‘ladi. `<identity12>` — model identifikatori (`MANIFEST.json` da qayd etilgan ochiq matn SHA-256 ining dastlabki 12 ta hex belgisi); `<fp8>` *Sozlamalar → Litsenziya* da ko‘rsatilgan barmoq izining dastlabki 8 belgisiga teng bo‘lishi shart.

Dastlabki shartlar: siz **admin** sifatida kirgansiz; xuddi shu papkadagi `license.dat` o‘rnatilgan (5.3-band), ilova qayta ishga tushirilgan va *Sozlamalar → Litsenziya* da *Litsenziyalangan* ko‘rsatilgan.

1. Zip faylni USB tashuvchidan kompyuterga nusxalang. Uni **ochmang** (arxivdan chiqarmang).
2. *Sozlamalar → AI modellari → Model paketini import qilish* → zip faylni tanlang. Ilova uni `<ma’lumotlar papkasi>\model_inbox\` ga nusxalaydi va AI serveridan uni o‘rnatishni so‘raydi (`POST /models/import`, faqat administrator).
3. Server qayta ishga tushirilmasdan: fayl `model_inbox\` ichida ekanini tekshiradi; uni vaqtinchalik papkaga ochadi; `MANIFEST.json` dagi ochiq matn (`model.safetensors`) va shifrlangan matn (`model.safetensors.enc`) yozuvlarini hamda yo‘ldosh fayldagi `plaintext_sha256` ni tekshiradi; **og‘irliklarni shu kompyuter kaliti bilan sinov tariqasida deshifrlaydi**; joriy `brain_triage_finetuned\` ni `brain_triage_finetuned_prev_<eski identifikator>` ga ko‘chiradi; yangi papkani modellar papkasiga (3-band) joylaydi va detektorni qayta yuklaydi. Ochiq og‘irliklar diskka yozilmaydi. Qadam audit jurnaliga yoziladi.
4. Bildirishnoma yangi **identifikator**ni ko‘rsatadi. U fayl nomidagi `<identity12>` ga va yetkazib beruvchi README sidagi identifikatorga teng bo‘lishi shart. So‘ngra tekshiring: `/health` → `"status": "ok"` va `models.brain_triage.loaded: true`, *Sozlamalar → AI modellari* da *Tekshiruv triaji* modeli xuddi shu identifikator bilan ko‘rsatilgan, bitta sinov tekshiruvidan keyin u PDF pastida ham chop etilgan. Identifikator va sanani klinikaning o‘zgarishlar jurnaliga yozing.

Biror narsa bajarilmasa, oldingi model joyida qoladi (avtomatik orqaga qaytarish), bildirishnoma esa sababni aytadi:

| Sabab kodi | Ma’nosi | Nima qilish kerak |
|---|---|---|
| `bundle_not_for_this_machine` | Paket boshqa kompyuter uchun shifrlangan yoki o‘rnatilgan litsenziya xuddi shu yetkazib berishdan emas (kalit mos kelmaydi). | Fayl nomidagi `<fp8>` ni *Sozlamalar → Litsenziya* bilan solishtiring; xuddi shu papkadagi litsenziyani import qilib, qayta ishga tushiring; aks holda yangi paket uchun joriy barmoq izini yetkazib beruvchiga yuboring. |
| `license_model_key_missing` | Ushbu kompyuterda model kalitli faol litsenziya yo‘q: yetkazib berishdagi `license.dat` hali import qilinmagan yoki import qilingandan keyin ilova qayta ishga tushirilmagan. | Xuddi shu papkadagi `license.dat` ni import qiling (5.3-band), ilovani qayta ishga tushiring va paketni yana import qiling. |
| `manifest_mismatch`, `sidecar_mismatch`, `decrypted_hash_mismatch`, `bundle_incomplete`, `bundle_invalid` | Fayl ko‘chirishda buzilgan (USB, to‘liq bo‘lmagan nusxa), o‘zgartirilgan yoki to‘liq emas. | USB tashuvchidan qaytadan nusxalang; `certutil -hashfile <zip> SHA256` natijasini yetkazib beruvchining ilova xatidagi qiymat bilan solishtiring. |
| `path_outside_inbox`, `bundle_not_zip`, `bundle_contains_plaintext` | Fayl *Model paketini import qilish* orqali berilmagan yoki u shifrlangan Sentinel model paketi emas. | *Sozlamalar → AI modellari* dagi tugmadan foydalaning; faqat yetkazib beruvchi papkasidagi zip ni import qiling. |
| `models_dir_not_writable`, `model_dir_overridden` | Dasturning modellar papkasiga yozib bo‘lmaydi (ilova barcha foydalanuvchilar uchun Program Files ga o‘rnatilgan) yoki `SENTINEL_MODEL_DIR_OVERRIDE_BRAIN_TRIAGE` berilgan. | Joriy foydalanuvchi uchun qayta o‘rnating (o‘rnatuvchining standart varianti) yoki import uchun ilovani bir marta *Administrator nomidan ishga tushirish* orqali oching; almashtirish o‘zgaruvchisini olib tashlang. |
| HTTP 403 / `import_in_progress` | Kirgan foydalanuvchi administrator emas / boshqa import hali bajarilmoqda. | admin sifatida kiring / kuting va qaytadan urining. |

Olingan papkani saqlang (USB tashuvchi va klinikaning zaxira nusxalari bilan birga nusxa, 6.1-band): **ilovaning har bir yangilanishidan keyin xuddi shu paketni qayta import qiling** (8.1-band) — yangilanish import qilingan model joylashgan dastur papkasini almashtiradi.

### 5.5 Tasdiqlanmagan (pending) detektorlar — standart holatda o‘chiq

Ommaviy *bosh miya o‘smasi* klassifikatori **tasdiqlash kutilmoqda** holatida va mahalliy tomograflarda o‘smalarni ortiqcha belgilaydi. Shu sababli 1.1.0 versiyasidan boshlab kasalxona yig‘ilmalari **pending holatidagi detektorlarni ishga tushirmaydi**; shifokorlar faqat tasdiqlangan *tekshiruv triaji* natijasini ko‘radi. Eksperimental detektorlar butun tekshiruv tahlilida hech qachon ishga tushirilmaydi.

- Kalit: *Sozlamalar → AI modellari → Tasdiqlanmagan (pending) detektorlarni ko‘rsatish* (faqat administratorlar). O‘zgarish qayta ishga tushirishsiz keyingi tahlildan kuchga kiradi, `<ma’lumotlar papkasi>\settings.json` ga saqlanadi va audit jurnaliga yoziladi (kim, qachon, eski → yangi qiymat). Kalit yoqilgan paytda AI modellari sahifasida ogohlantiruvchi belgi ko‘rinadi. API: administrator tokeni bilan `GET` / `PUT /settings/ai`; `/health` → `ai_settings` joriy qiymatni va uning manbasini (`file`, `env` yoki `default`) ko‘rsatadi.
- Boshlang‘ich qiymatni `SENTINEL_SHOW_PENDING_DETECTORS` (9-band) bilan berish mumkin; ilovada saqlangan qiymat ustun turadi.
- Uni faqat klinikaning klinik rahbari yozma rozilik bergandan keyin yoqing (masalan, pilot davomida tadqiqot taqqoslashi uchun) va shifokorlarni ogohlantiring.
- Ushbu relizda **bosh KT** uchun tasdiqlangan detektor yo‘q: kalit o‘chiq bo‘lsa, bosh KT tekshiruvi ish ro‘yxatiga qabul qilinadi, lekin *Ko‘rib chiqish kerak* holati va u uchun tasdiqlangan AI detektori yo‘qligi haqidagi izoh bilan qaytariladi — rentgenolog uni odatdagidek o‘qiydi. Kalit yoqilgan bo‘lsa, ommaviy qon quyilishi klassifikatori ishga tushadi va uning natijasi *tasdiqlash kutilmoqda* deb belgilanadi.

---

## 6. Ma’lumotlar papkasi

Barcha o‘zgaruvchan holat **bitta papkada** joylashadi — hech qachon dastur papkasi ichida emas:

| Platforma | Standart joylashuv |
|---|---|
| Windows | `%APPDATA%\sentinel-medical-ai\` |
| macOS | `~/Library/Application Support/sentinel-medical-ai/` |
| Linux | `~/.local/share/sentinel-medical-ai/` |

Yig‘ilgan ish stoli ilovasi o‘zining foydalanuvchi ma’lumotlari papkasini serverga `SENTINEL_DATA_DIR` orqali uzatadi. **Ishonchli yo‘lni server xabar qiladi**: `GET /health` → `data_dir`, hamda *Sozlamalar → Ulanishlar → Server ma’lumotlar papkasi*. Shubha bo‘lsa, ushbu jadvaldan emas, o‘sha yo‘ldan foydalaning.

Tarkibi:

| Yo‘l | Nima | Shaxsiy tibbiy ma’lumot (PHI)? |
|---|---|---|
| `sentinel.db` (+ `-wal`, `-shm`) | SQLite bazasi: foydalanuvchilar (bcrypt xeshlari), tekshiruvlar (DICOM sarlavha maydonlari, shu jumladan bemor ID/ismi/tug‘ilgan sanasi), AI natijalari, **imzolangan hisobotlar** (matn, imzolagan shaxs, vaqt, SHA-256, model identifikatori), tuzatishlar, **audit jurnali** | **Ha** |
| `logs\sentinel_YYYY-MM-DD.log` | Server jurnali, 10 MB da aylantiriladi, zip qilinadi, 30 kun saqlanadi. Bemor identifikatorlari yashiriladi (7-band). | Yashirilgan |
| `logs\backend.log` | Ish stoli qobig‘i tomonidan ushlab olingan serverning xom konsol chiqishi | Yashirilgan |
| `doctor_corrections\` | Rentgenolog shifokorlarning AI qoralamalariga tahrirlari, faqat qo‘shiladigan JSONL (hisobot matni, tekshiruv ID, foydalanuvchi ID) | Hisobot matni |
| `training_corpus\` | O‘quv ma’lumotlari yig‘uvchisi yozadigan **identifikatorlarsiz** miniatyuralar va yorliqlar (ismlar, ID lar, tug‘ilgan sanalar yo‘q) | Identifikatorlarsiz |
| `dicom_cache\` | Vaqtinchalik DICOM keshi | Ha |
| `license.dat` | Kompyuterga bog‘langan litsenziya. Klinika variantida model kalitining sirini ham saqlaydi (5.3-band) — uni maxfiy deb hisoblang va yetkazib beruvchidan boshqa hech kimga bermang. | Yo‘q |
| `jwt_secret.key` | Kirish tokenlarini imzolaydigan sir (rejim 0600). Uni o‘chirish hammani tizimdan chiqaradi. | Yo‘q |
| `model_key.bin` | Litsenziyada kalit siri bo‘lmaganda model kaliti uchun o‘rnatishga xos sir (to‘liq variant, 8.3-band). Birinchi foydalanishda yaratiladi, rejim 0600. Yo‘qolsa, shifrlangan og‘irliklarni o‘qib bo‘lmaydi. | Yo‘q |
| `settings.json` | Administrator saqlagan AI sozlamalari (*Tasdiqlanmagan (pending) detektorlarni ko‘rsatish*, 5.5-band) | Yo‘q |
| `model_inbox\` | *Sozlamalar → AI modellari → Model paketini import qilish* orqali import qilingan model paketlari (5.4-band) | Yo‘q |

Yuklangan DICOM fayllari tahlil paytida server tanlagan nomlar bilan vaqtinchalik papkaga yoziladi va darhol o‘chiriladi; server sarlavha maydonlari va kichik oldindan ko‘rish tasvirini saqlaydi, tekshiruvning o‘zini emas.

### 6.1 Zaxira nusxa

Ma’lumotlar bazasi **shifrlanmagan holda** saqlanadi. Kompyuterni shifrlangan diskda ishlating (BitLocker / FileVault) va zaxira nusxani quyidagicha oling:

1. Ish stoli ilovasini yoping (bu serverni to‘xtatadi) yoki hech qanday tahlil bajarilmayotganiga ishonch hosil qiling.
2. **Butun ma’lumotlar papkasini** (6-band) tashqi diskka yoki NAS ga nusxalang — kamida `sentinel.db`, `sentinel.db-wal`, `sentinel.db-shm`, `doctor_corrections\`, `license.dat`, `jwt_secret.key`, `model_key.bin`, `settings.json`, `model_inbox\`, `logs\`. `license.dat` / `model_key.bin` siz tiklangan o‘rnatish o‘z modelini deshifrlay olmaydi.
3. Uchta nusxa saqlang (mahalliy, tashqi, boshqa joyda), olinadigan tashuvchilarni shifrlang va zaxira kompyuterda har oy **tiklashni sinab ko‘ring**: xuddi shu versiyani o‘rnating, papkani qaytarib nusxalang, ishga tushiring, tizimga kiring, imzolangan hisobotni oching.

Yordamchi skript mavjud (`scripts\backup_restore.py backup --out <papka>`, nazorat summali `.tar.gz` yaratadi), lekin u bazani backend ning `data\` papkasi ostida kutadi; unga tayanishdan oldin u haqiqatan ham sizning ma’lumotlar papkangizni olganini tekshiring. Yuqoridagi oddiy papka nusxasi har doim to‘g‘ri.

### 6.2 Dasturni o‘chirish

Windows o‘chirish dasturi ma’lumotlar papkasini o‘chirish-o‘chirmaslikni so‘raydi. Klinika yozuvlarni yo‘q qilishga qaror qilmagan bo‘lsa, **Yo‘q** deb javob bering: imzolangan hisobotlar va audit jurnali — klinikaning tibbiy-huquqiy arxivi.

---

## 7. Jurnallar va identifikatorlarni yashirish

- Server jurnali (`logs\sentinel_*.log`) tekshiruv ID larini (server yaratgan UUID lar), marshrutlarni, bajarilish vaqtlarini, modellar yuklanish holatini, foydalanuvchi nomi bo‘yicha kirish urinishlarini va xatolarni qayd etadi. Global filtr har qanday qator yozilishidan oldin aniq bemor identifikatorlarini (`PatientID=…`, `PatientName=…`, tug‘ilgan sanalar va `Familiya^Ism` ko‘rinishidagi DICOM ismlari) `<redacted>` bilan almashtiradi. Mijoz fayl nomlari hech qachon jurnalga yozilmaydi.
- **Audit jurnali** (baza jadvali, shuningdek administratorlar uchun `GET /audit/log`, oxirgi 200 ta yozuv) kim, nima va qachon qilganini qayd etadi: `login`, `analyze_study`, `view_study`, `sign_report`, `change_password`, `register`, mijoz IP manzili va hisobot xeshi kabi tafsilotlar bilan.
- Audit jurnali — **xesh zanjiri**: har bir qator oldingi qatorning xeshini (`prev_hash`) va o‘zining xeshini (`row_hash`) saqlaydi, shuning uchun o‘chirilgan yoki tahrirlangan qator zanjirni buzadi. `GET /audit/verify` (faqat administrator) zanjirni boshidan oxirigacha tekshiradi va `{ok, rows, checked, head_hash, first_broken}` ni qaytaradi. Uni zaxira nusxadan tiklagandan keyin va bazani biror kishiga topshirishdan oldin bajaring; `ok: false` bo‘lsa, `first_broken` dagi qator identifikatorini qayd eting va yetkazib beruvchiga xabar bering.
- Ish stoli ilovasining o‘zi faqat o‘z konsoliga yozadi (dasturchi asboblari faqat dasturchi yig‘ilmalarida ochiq).

Jurnallarni yetkazib beruvchiga yuborishda faqat `sentinel_*.log` / `backend.log` ni yuboring; ma’lumotlar bazasini — hech qachon. *Diagnostikani eksport qilish* (10-band) afzal: qo‘lda nusxalangan jurnalda birinchi ishga tushirishdagi bir martalik administrator paroli (5-band) qolishi mumkin, eksport uni olib tashlaydi.

---

## 8. Ilova va modellarni yangilash

### 8.1 Ilovani yangilash

1. Ma’lumotlar papkasining zaxira nusxasini oling (6.1-band).
2. Yangi o‘rnatuvchini eski versiya ustidan ishga tushiring (Windows) yoki ilova paketini almashtiring (macOS). Ma’lumotlar papkasiga tegilmaydi.
3. Ilovani bir marta administrator sifatida ishga tushiring. Ma’lumotlar bazasi **o‘zi migratsiya qiladi** (yangi ustunlar qo‘shiladi; hech narsa o‘chirilmaydi). `/health` → `version` ni va kirish ekranidagi versiyani tekshiring.
4. Ish stoli ilovasi va server bir xil versiya raqamiga ega (ushbu tahrirda 1.1.0); versiya har bir PDF da chop etiladi.
5. **Klinika varianti:** yangilanish dastur papkasini import qilingan model bilan birga almashtiradi — xuddi shu model paketini qayta import qiling (5.4-band) va `/health` `"status": "ok"` qaytarishini tekshiring. To‘liq variant o‘rnatuvchisi modelni o‘zi bilan olib keladi va birinchi ishga tushirishda uni yana shifrlaydi (8.3-band).

### 8.2 Modelni yangilash — MANIFEST va reliz darvozasi

Tasdiqlangan detektor — bu papka: `models\brain_triage_finetuned\`, tarkibida `model.safetensors`, `config.json`, `preprocessor_config.json`, **`MANIFEST.json`** va `study_level_eval.json`.

- `MANIFEST.json` **har bir og‘irlik faylining SHA-256** xeshini qayd etadi. `model.safetensors` xeshining dastlabki 12 ta hex belgisi — ilova model barmoq izi sifatida ko‘rsatadigan qiymat («Tafsilotlar» yorlig‘i, Topilmalar → Modellar, har bir PDF). Joriy reliz: `0d559766ce58…`.
- Yetkazib beruvchi modelni jo‘natishdan oldin u **reliz darvozasidan** o‘tishi shart: `scripts\eval_study_level.py` manifestni tekshiradi, modelni ajratib qo‘yilgan mahalliy validatsiya to‘plamida tekshiruv-ba-tekshiruv baholaydi va ishlab chiqarish ish nuqtasida (5 ta markaziy kesim bo‘yicha o‘rtacha, chegara 0,50) tekshiruv darajasidagi sezgirlik < 0,85 yoki spetsifiklik < 0,40 bo‘lsa relizni rad etadi (nolga teng bo‘lmagan chiqish kodi). Hosil bo‘lgan `study_level_eval.json` model bilan birga yetkaziladi.

Model yangilanishini o‘rnatish:

1. Ilovani to‘xtating.
2. Butun `brain_triage_finetuned\` papkasini yetkazib beruvchi yuborgan papka bilan almashtiring (eskisining nusxasini saqlang).
3. Xeshlarni o‘zingiz tekshiring: Windows `certutil -hashfile model.safetensors SHA256`, macOS `shasum -a 256 model.safetensors`; `MANIFEST.json` bilan solishtiring.
4. Ilovani ishga tushiring; `/health` → `models.brain_triage.loaded = true` ekanini va bitta sinov tekshiruvidan keyin «Tafsilotlar» yorlig‘idagi barmoq izi manifest xeshiga teng ekanini tekshiring.
5. Model barmoq izini va sanani klinikaning o‘zgarishlar jurnaliga yozing; shu paytdan boshlab imzolangan hisobotlar yangi identifikatorni oladi.

Model papkasi ichidagi fayllarni hech qachon tahrirlamang va nomini o‘zgartirmang: server identifikatori tasdiqlanganiga mos kelmaydigan modelni yuklaydi.

### 8.3 Shifrlangan model og‘irliklari — birinchi ishga tushirishda avtomatik

Mahalliy o‘qitilgan detektor — yetkazib beruvchining intellektual mulki; har bir klinika kompyuterida u **diskda shifrlangan holda** (AES-256-GCM) saqlanadi. Shifrlangan model papkasida `model.safetensors` o‘rniga `model.safetensors.enc` va `model.safetensors.enc.meta.json` bo‘ladi. Server og‘irliklarni yuklash paytida to‘g‘ridan-to‘g‘ri xotiraga deshifrlaydi; ochiq matn o‘rnatish diskiga hech qachon yozilmaydi.

- **Avtomatik shifrlash (to‘liq variant).** O‘rnatilgan Windows ilovasi serverni `SENTINEL_ENCRYPT_MODELS_AT_REST=1` bilan ishga tushiradi. Birinchi ishga tushirishda server har bir xususiy `*_finetuned` modelning ochiq og‘irliklarini joyida shu o‘rnatish kaliti bilan shifrlaydi, natijani qayta deshifrlab, ochiq matnni o‘chirishdan **oldin** uning SHA-256 ini ochiq matn bilan solishtiradi va har qanday xatoda asl holatni tiklaydi. O‘rnatish kaliti bilan allaqachon shifrlangan model, agar kalit endi litsenziyadan olinsa, xuddi shu tarzda qayta shifrlanadi. `sentinel_*.log` dagi qator (*Encryption at rest: OK …* yoki *ENCRYPTION AT REST FAILED …*) natijani har ikki holatda ham bildiradi; xatoda ham server ishga tushadi. O‘zgaruvchi berilmagan bo‘lsa (dasturchi ishga tushirishi, CI tutun testlari, manba koddan o‘rnatish), `SENTINEL_DEV_INSECURE=1` bo‘lsa va macOS da imzolangan `.app` paketi ichida hech narsa qilinmaydi (bunday DMG modelni shifrlanmagan holda saqlaydi — klinikalar uchun Windows dan foydalaning). Model identifikatori o‘zgarmaydi: u ochiq matnning SHA-256 i.
- **Klinika varianti.** Paket shu kompyuter uchun allaqachon shifrlangan holda keladi (5.4-band); ishga tushirishda hech narsa shifrlanmaydi.
- **Kalit** quyidagi tartibda tanlanadi: `SENTINEL_MODEL_KEY_HEX` (xom kalit, faqat yetkazib beruvchi uchun) → **amaldagi** litsenziya ichidagi `model_key_secret` (klinika varianti, 5.3-band) → ma’lumotlar papkasidagi `model_key.bin` (birinchi foydalanishda yaratiladi, rejim 0600). Oxirgi ikkitasi kompyuter barmoq izi bilan birlashtiriladi. `python -m src.utils.model_crypto keyinfo` qaysi manba ishlatilganini ko‘rsatadi. Litsenziya muddati tugasa yoki kuchini yo‘qotsa, `model_key.bin` yo‘qolsa yoki barmoq izi o‘zgarsa, og‘irliklarni deshifrlab bo‘lmaydi — `/health` `degraded` holatini ko‘rsatadi; yetkazib beruvchiga murojaat qiling.
- **Xeshni tekshirish** 8.2-banddagi 3-qadamdan farq qiladi: `MANIFEST.json` **ochiq matn** SHA-256 ni (`model.safetensors` yozuvi — ilova ko‘rsatadigan barmoq izi) va **shifrlangan fayl** SHA-256 ni (`model.safetensors.enc` yozuvi) qayd etadi. `certutil -hashfile model.safetensors.enc SHA256` (macOS: `shasum -a 256 model.safetensors.enc`) natijasini `.enc` yozuvi bilan, `model.safetensors.enc.meta.json` ichidagi `plaintext_sha256` ni esa `model.safetensors` yozuvi bilan solishtiring.
- Buyruq qatori vositasi (backend papkasidan): `python -m src.utils.model_crypto encrypt <model_papkasi>` (og‘irliklar → `.enc`, ochiq matn o‘chiriladi), `python -m src.utils.model_crypto decrypt <model_papkasi>` (teskari amal) va `python -m src.utils.model_crypto keyinfo` (kalit manbai va kalit barmoq izi; kalitning o‘zi emas). `encrypt`/`decrypt` — yetkazib beruvchining reliz tayyorlash qadamlari; ularni hech qachon ishlayotgan `models\` daraxtida bajarmang, faqat nusxada. Klinika kompyuterida odatda faqat `keyinfo` kerak bo‘ladi — yetkazib beruvchi so‘raganda.

---

## 9. Muhit o‘zgaruvchilari

Server jarayoni uchun beriladi (yig‘ilgan distributivda ish stoli qobig‘i `SENTINEL_REQUIRE_AUTH=1`, `SENTINEL_DATA_DIR`, `SENTINEL_HOSPITAL_BUILD=1` va `SENTINEL_ENCRYPT_MODELS_AT_REST=1` ni o‘zi o‘rnatadi).

| O‘zgaruvchi | Standart qiymat | Vazifasi |
|---|---|---|
| `SENTINEL_DATA_DIR` | platformaning foydalanuvchi ma’lumotlari papkasi (6-band) | Baza, jurnallar, tuzatishlar, litsenziya va JWT siri qayerda joylashadi |
| `SENTINEL_MODELS_DIR` | `<backend>\models` | Faqat o‘qish uchun modellar to‘plami |
| `SENTINEL_ADMIN_PASSWORD` | berilmagan (tasodifiy bir martalik parol) | Birinchi ishga tushirishda yaratiladigan `admin` hisob qaydnomasining paroli; har ikkala holda ham «o‘zgartirish shart» deb belgilanadi |
| `SENTINEL_REQUIRE_AUTH` | standart holatda yoqilgan | `1` autentifikatsiyani dasturchi rejimida ham majburan yoqadi |
| `SENTINEL_DEV_INSECURE` | berilmagan | `1` = **faqat dasturchi rejimi**: autentifikatsiyani bo‘shashtiradi, `/docs` ni ochadi, modellarni tarmoqdan yuklashga ruxsat beradi. Klinika kompyuterida hech qachon. |
| `DEV_BYPASS_LICENSE` | berilmagan | `1` litsenziya tekshiruvini o‘tkazib yuboradi — **faqat** `SENTINEL_DEV_INSECURE=1` bilan birga hisobga olinadi |
| `SENTINEL_HOSPITAL_BUILD` | berilmagan | `1` kasalxona yig‘ilmasini bildiradi: oflayn rejim dasturchi rejimida ham standart bo‘ladi (uni faqat aniq berilgan `SENTINEL_OFFLINE=0` bekor qiladi) |
| `SENTINEL_OFFLINE` | dasturchi rejimidan tashqari `1` | `0` modellarni yuklab olishga ruxsat beradi (faqat dasturchi kompyuterlari) |
| `SENTINEL_JWT_SECRET` | `jwt_secret.key` ga yaratiladi | Token imzolash sirini almashtirish |
| `SENTINEL_SKIP_WARMUP` | berilmagan | `1` ishga tushirishda bosh miya modellarini yuklashni o‘tkazib yuboradi (testlar); `/health` ularni birinchi foydalanishgacha yuklanmagan deb ko‘rsatadi |
| `SENTINEL_ORTHANC` / `ORTHANC_URL` | o‘chiq / `http://localhost:8042` | `1` ixtiyoriy Orthanc PACS kuzatuvchisini yoqadi |
| `SENTINEL_ALLOW_CLOUD_LLM` | berilmagan | `1` bulutli LLM backend ga ruxsat beradi. Klinikada bermang. |
| `SENTINEL_CORS_ORIGINS` / `SENTINEL_CORS_ALLOW_ALL` | localhost + Electron manbalari / o‘chiq | Qo‘shimcha brauzer manbalari; `ALLOW_ALL=1` ishlab chiqarish uchun emas |
| `SENTINEL_TLS_CERT`, `SENTINEL_TLS_KEY` | berilmagan | HTTPS orqali xizmat ko‘rsatish — server qachondir localhost dan tashqariga ochilsa, majburiy |
| `SENTINEL_WORKERS` | `1` | Uvicorn ishchi jarayonlari soni |
| `SENTINEL_MODEL_KEY_HEX` | berilmagan: kalit litsenziyadagi `model_key_secret` dan, aks holda `model_key.bin` dan (+ kompyuter barmoq izi) | Shifrlangan model og‘irliklari uchun xom AES kaliti; boshqa ikkala manbadan ustun (faqat yetkazib beruvchi uchun; 8.3-band) |
| `SENTINEL_ENCRYPT_MODELS_AT_REST` | o‘rnatilgan Windows/macOS ilovasida `1` (qobiq o‘rnatadi), aks holda `0` yoki berilmagan | `1` = ishga tushirishda xususiy modellarning ochiq og‘irliklarini joyida shifrlash (o‘chirishdan oldin tekshirish, xatoda orqaga qaytarish; 8.3-band). `SENTINEL_DEV_INSECURE=1` bilan va macOS `.app` paketi ichida e’tiborga olinmaydi. Qo‘lda bermang. |
| `SENTINEL_SHOW_PENDING_DETECTORS` | kasalxona yig‘ilmalarida (`SENTINEL_HOSPITAL_BUILD=1`) `0`, aks holda `1` | *Tasdiqlanmagan (pending) detektorlarni ko‘rsatish* ning boshlang‘ich qiymati (5.5-band); `1` pending detektorlarni, masalan ommaviy o‘sma klassifikatorini ishga tushiradi. *Sozlamalar → AI modellari* da saqlangan qiymat (`settings.json`) ustun turadi. |
| `SENTINEL_MODEL_DIR_OVERRIDE_<KEY>` | berilmagan | Bitta detektorni boshqa papkaga yo‘naltirish (faqat yetkazib beruvchining A/B sinovlari) |
| `SENTINEL_DEFAULT_LANG` | `ru` | Ixtiyoriy papka kuzatuvchisi uchun hisobot tili |
| `OLLAMA_KEEP_ALIVE` (Ollama ning o‘z o‘zgaruvchisi) | — | masalan `24h` — Gemma ni xotirada ushlab turadi, kunning birinchi hisoboti sekin bo‘lmasligi uchun |

---

## 10. Nosozliklarni bartaraf etish

**`GET http://127.0.0.1:8000/health`** dan boshlang (kompyuterdagi brauzerda oching; kirish talab qilinmaydi). U quyidagini qaytaradi:

```json
{ "status": "ok" | "degraded" | "error", "version": "1.1.0", "device": "cpu",
  "auth_required": true, "offline": true,
  "ai_settings": { "show_pending_detectors": false, "source": "default" },
  "models": { "brain_triage": {"loaded": true, "reason": "…", "identity": "<12 hex>"},
              "brain_tumor_class": {"loaded": false, "reason": "not loaded"},
              "chest": {"loaded": false, "reason": "not loaded"} },
  "llm": { "backend": "ollama" | "template" | null, "reachable": true },
  "license": { "mode": "licensed" | "unlicensed" | "demo" | "dev" },
  "data_dir": "…", "uptime_seconds": 123.4 }
```

| Belgi | Qayerga qarash | Yechim |
|---|---|---|
| Kirish ekranida **AI serveri: ishlamayapti**; qizil banner *AI serveri ishlamayapti* | `/health` javob bermaydi. `<ma’lumotlar papkasi>\logs\backend.log` va eng yangi `sentinel_*.log` | `resources\backend\run_server.py` va uning `venv` papkasi mavjudligini; 8000 portni boshqa dastur band qilmaganini tekshiring; jurnaldagi oxirgi traceback ni o‘qing. Ish stoli qobig‘i serverni 3 marta qayta ishga tushiradi (2 s / 4 s / 8 s), so‘ngra jurnal yo‘li bilan dialog ko‘rsatadi. Ilovani qayta ishga tushiring; davom etsa, kompyuterni qayta yoqing; keyin jurnal bilan yetkazib beruvchiga murojaat qiling. |
| Banner *AI serveri qisman ishlamoqda: tasdiqlangan model yuklanmadi* | `/health` → `status: degraded`, `models.brain_triage.loaded: false`, `reason` ni o‘qing | **Klinika varianti, yangi o‘rnatish yoki yangilanishdan keyin:** model paketini import qiling (5.4-band). **Litsenziya muddati tugagan / apparat almashtirilgan** (klinika varianti): model kaliti litsenziyadan olinadi — uni uzaytiring yoki yangi yetkazib berishni so‘rang (5.3-band). Aks holda odatda **model yo‘q**: `brain_triage_finetuned` papkasi yo‘q, to‘liq emas yoki ko‘chirilgan — uni o‘rnatish tashuvchisidan / yetkazib beruvchi paketidan tiklang va qayta ishga tushiring. Ungacha tasdiqlangan AI natijasi yo‘q — rentgenolog shifokorlarga AI ga tayanmaslikni ayting. |
| `models.brain_tumor_class.loaded: false` | `ai_settings.show_pending_detectors`; `reason` qidirilgan papkalarni ko‘rsatadi | Tasdiqlanmagan detektorlar o‘chirilgan paytda bu **kutilgan holat** (5.5-band): model ishga tushishda yuklanmaydi va administrator ularni yoqqandan keyingi birinchi tekshiruvda yuklanadi. Agar ular yoqilgan bo‘lsa-yu xato qolsa, model to‘plamini (`models\hf\…`) o‘rnatish tashuvchisidan nusxalang. Halokatli emas: triaj detektori ishlashda davom etadi. `models.brain_triage.identity` — yuklangan triaj modelining 12 belgili identifikatori; uni yetkazib berish README fayli bilan solishtiring. |
| Holat panelida **LLM: mavjud emas**; bildirishnoma *Shablon hisobot — AI yordamchisi mavjud emas* | `/health` → `llm.reachable: false` | Ollama ishlamayapti yoki Gemma modeli yo‘q: `ollama list`, `ollama serve`, `ollama pull gemma3:4b`. Topilmalar va imzolashga ta’sir qilmaydi. |
| *Sozlamalar → Litsenziya* da **Litsenziyasiz** / **Demo rejim**; tahlillar «Demo limit reached (10 analyses/day)» bilan to‘xtaydi | `/license/status` → `info.reason` | `no license file at …` → `license.dat` ni ma’lumotlar papkasiga nusxalang; `invalid signature` → fayl buzilgan yoki tahrirlangan, yangi nusxa oling; `bound to a different machine` → apparat/xost nomi o‘zgargan, yangi barmoq izini yetkazib beruvchiga yuboring; `license expired on …` → uzaytiring. Faylni almashtirgandan keyin qayta ishga tushiring. |
| Shifokor imzolay olmaydi: *Sizning rolingiz hisobot imzolashga ruxsat bermaydi* | `/auth/me` dagi rol | Faqat *radiologist* va *admin* imzolay oladi; laborantlar — yo‘q. Foydalanuvchini to‘g‘ri rol bilan qayta yarating. |
| *Bu tildagi hisobot allaqachon imzolangan* (HTTP 409) | kutilgan xatti-harakat | Har bir tekshiruv va til uchun bitta imzolangan hisobot. Xato emas. |
| Tekshiruv **Ko‘rib chiqish kerak** bilan qaytadi | «Topilmalar» yorlig‘i sababini ko‘rsatadi | Bosh miyaga oid bo‘lmagan MRT, modeli o‘rnatilmagan modallik yoki — ushbu relizda bosh KT — pending detektorlar o‘chiq bo‘lganda *tasdiqlangan* detektor yo‘q (5.5-band; izoh shuni aytadi). Tekshiruv AI siz o‘qilishi kerak. Nosozlik emas. |
| Model importi rad etildi | *Model paketini import qilish* dan keyingi bildirishnoma; audit jurnali | 5.4-banddagi sabablar jadvaliga qarang (`bundle_not_for_this_machine`: litsenziya va paket shu kompyuter uchun bitta yetkazib berishdan emas). Oldingi model joyida qoladi. |
| Yuklash rad etildi: *No readable DICOM file in the upload* | — | Fayllar DICOM emas (skrinshotlar, ZIP, NIfTI). Tekshiruvni tomograf/PACS dan DICOM sifatida eksport qiling. |
| Hamma bir vaqtda tizimdan chiqib ketdi | `jwt_secret.key` o‘chirilgan yoki `SENTINEL_JWT_SECRET` o‘zgartirilgan | Bunday o‘zgarishdan keyin kutilgan holat; foydalanuvchilar shunchaki qayta kiradi. |
| Administrator paroli unutilgan | — | Tiklash interfeysi yo‘q. Serverni to‘xtating, `sentinel.db` ning zaxira nusxasini oling va tiklash tartibi uchun yetkazib beruvchiga murojaat qiling (yangi bir martalik administrator paroli faqat administrator hisob qaydnomasi mavjud bo‘lmaganda yaratiladi). |
| PDF eksporti bajarilmaydi | «Hisobot» yorlig‘idagi bildirishnoma *PDF eksport qilib bo‘lmadi* | Bo‘sh disk joyini va tanlangan papkaga yozish mumkinligini tekshiring; «Yuklab olishlar» papkasini sinab ko‘ring. |

Server uchun pytest tekshiruv testlarini (`tests\`) so‘rov bo‘yicha yetkazib beruvchi bajarishi mumkin.

**Diagnostikani eksport qilish.** Qo‘llab-quvvatlash so‘rovi uchun *Sozlamalar → Dastur haqida → Diagnostikani eksport qilish* dan foydalaning (faqat administratorlar; ilova `GET /diagnostics` ni chaqiradi va `sentinel-diagnostics-<YYYYMMDD-HHMM>.zip` ni qayerga saqlashni so‘raydi). Jurnal fayllarini qo‘lda tanlash o‘rniga yetkazib beruvchiga shu zip ni yuboring. Uning tarkibi:

- `health.json` (`/health` javobi), `license.json` (faqat litsenziya holati — litsenziya faylining o‘zi hech qachon emas), `ai_settings.json`;
- `versions.json` (ilova, Python, PyTorch, transformers, operatsion tizim) va `models.json` (har bir modelning identifikatsiya kartasi: nomi, SHA-256 barmoq izi, tasdiqlash holati);
- `logs\backend.log` ning oxiri (so‘nggi 2 MB) va eng yangi `sentinel_*.log` ning oxiri — identifikatorlari allaqachon yashirilgan (7-band); parollar (birinchi ishga tushirishdagi bir martalik administrator paroli ham), tokenlar, API va litsenziya kalitlari, `Authorization` sarlavhalari va URL dagi user:password qismlari `[REDACTED]` bilan almashtirilgan;
- `audit_tail.jsonl` (so‘nggi 200 ta audit yozuvi, bemor ID va ismi maydonlari `<redacted>` bilan almashtirilgan) va `settings_redacted.json`.

Unda tasvirlar va prevyular, ma’lumotlar bazasi, `license.dat`, `jwt_secret.key`, `model_key.bin` va model fayllari **yo‘q**. Yuborishdan oldin zip ni ochib, ko‘zdan kechiring; u bemor ma’lumotlarisiz bo‘lishi uchun loyihalangan.

---

## 11. Foydalanuvchilarni boshqarish (administrator)

Ushbu yig‘ilmada foydalanuvchilarni boshqarish ekrani yo‘q; klinika kompyuteridan API dan foydalaning (PowerShell ko‘rsatilgan, `curl` xuddi shunday ishlaydi):

```powershell
# 1. admin sifatida kirish → token
$r = Invoke-RestMethod -Method Post http://127.0.0.1:8000/auth/login -ContentType application/json `
     -Body '{"username":"admin","password":"<administrator paroli>"}'
$h = @{ Authorization = "Bearer $($r.access_token)" }

# 2. rentgenolog shifokor yaratish (rollar: radiologist | technician | admin)
Invoke-RestMethod -Method Post http://127.0.0.1:8000/auth/register -Headers $h -ContentType application/json `
     -Body '{"username":"karimova","password":"<vaqtinchalik parol>","fullName":"Karimova N. A.","role":"radiologist"}'

# 3. foydalanuvchilar ro‘yxati / audit jurnalini o‘qish
Invoke-RestMethod http://127.0.0.1:8000/auth/users -Headers $h
Invoke-RestMethod "http://127.0.0.1:8000/audit/log?limit=200" -Headers $h

# 4. audit jurnali xesh zanjirini tekshirish (ok: true kutiladi)
Invoke-RestMethod http://127.0.0.1:8000/audit/verify -Headers $h
```

Har bir shifokorga shaxsiy hisob qaydnomasi bering — hisobotdagi imzo bu uni imzolagan hisob qaydnomasi. Parollar bcrypt xeshlari sifatida saqlanadi; API ularni qaytarib ko‘rsata olmaydi.

---

## 12. Xavfsizlik bo‘yicha eslatmalar

- **Autentifikatsiya standart holatda yoqilgan** va `SENTINEL_DEV_INSECURE=1` siz o‘chirilmaydi; bu o‘zgaruvchini klinika kompyuterida hech qachon bermaslik kerak. Klinik marshrutlar (`/analyze/study`, `/studies`, `/study/{id}`, `/report/sign`, `/report/save_correction`) bearer token talab qiladi; rollar server tomonida tekshiriladi. `/health` va `/license/status` ochiq va bemor ma’lumotlarini o‘z ichiga olmaydi.
- **Bulut yo‘q.** Server hech qachon internetga murojaat qilmaydi: modellar uchun oflayn rejim, bulutli LLM o‘chirilgan, Hugging Face telemetriyasi o‘chirilgan. Yagona tarmoq qo‘shnilari — xuddi shu kompyuterdagi ish stoli ilovasi va Ollama.
- **Portlar localhost ga bog‘langan** (`127.0.0.1:8000`, `localhost:11434`). Xostni `0.0.0.0` ga o‘zgartirmang va brandmauerga istisnolar qo‘shmang. Qachondir bir nechta kompyuterga joylashtirish talab qilinsa, TLS (`SENTINEL_TLS_*`) va yetkazib beruvchi bilan kelishilgan tarmoq sxemasidan foydalaning.
- API hujjatlari sahifalari (`/docs`, `/redoc`) ishlab chiqarish yig‘ilmalarida o‘chirilgan.
- **Diskni shifrlang** (BitLocker / FileVault): SQLite bazasi va jurnallar ilova tomonidan shifrlanmaydi. Yetkazib beruvchining xususiy model og‘irliklari esa shifrlanadi (8.3-band); ilova klinika variantidagi paketdan faqat u yaratilgan kompyuterda foydalanadi. E’tibor bering: `license.dat` zip-paket bilan birga istalgan kompyuterda og‘irliklarning kalitidir — yetkazib berish papkasini shifrlangan tashuvchida yoki yopiq zaxira nusxalarida saqlang, hech qachon umumiy papkada yoki elektron pochtada emas.
- Tasdiqlanmagan (pending) detektorlar standart holatda o‘chiq (5.5-band); bu sozlamaning har bir o‘zgarishi audit jurnaliga yoziladi. Diagnostika zip fayli (10-band) bemor ma’lumotlari, kalitlar va litsenziya faylini o‘z ichiga olmaydi.
- Imzolangan hisobotlar sezilmas o‘zgartirishdan himoyalangan (matn SHA-256 si, imzolagan shaxs, vaqt, model identifikatori, audit yozuvi), audit jurnali esa `GET /audit/verify` bilan tekshiriladigan xesh zanjiri (7-band) — lekin baza faylining o‘zini diskka kirish huquqi bo‘lgan har kim tahrirlashi mumkin; hisob qaydnomasi va diskni himoya qiling.
- Yuklangan fayllar diskda hech qachon mijoz fayl nomlari bilan saqlanmaydi; vaqtinchalik fayllar tahlildan keyin o‘chiriladi.
- Har bir foydalanuvchiga shaxsiy hisob qaydnomasi bering, ishdan ketgan xodimlarning hisob qaydnomalarini o‘chiring va bir martalik administrator parolini umumiy hujjatlarda saqlamang.

---

## 13. Windows o‘rnatuvchisini yig‘ish

3-bandda tasvirlangan o‘rnatuvchini GitHub Actions ish jarayoni `.github/workflows/build-windows.yml` (`SHAXZODBR/SIAA` repozitoriysi) yig‘adi. Avval `tests` topshirig‘i butun testlar to‘plamini Linux da faqat CPU uchun PyTorch bilan ishga tushiradi (`pytest -q -m "not slow"`, hujjatlar muvofiqligi tekshiruvi hamda ish stoli ilovasining tip tekshiruvi va modul testlari); o‘rnatuvchi topshiriqlari faqat u muvaffaqiyatli o‘tgandan keyin boshlanadi. So‘ngra har bir yig‘ish topshirig‘i Python serverini PyInstaller yordamida «muzlatadi» (`packaging/build_backend.ps1`), ochiq modellarni yuklab oladi, validatsiyadan o‘tgan triaj modelini yopiq modellar repozitoriysidan tiklaydi (identifikatori kodda mahkamlangan `brain-triage-<identity12>` relizi), muzlatilgan serverni `/health` orqali tekshiradi va so‘ngra electron-builder ni ishga tushiradi (`npm run build:win`). Xuddi shu ish jarayoni macOS-runnerda macOS uchun disk tasvirini (Apple Silicon) yig‘adi.

### 13.1 Yig‘ishni ishga tushirish va o‘rnatuvchini olish

1. GitHub → **Actions** → **Build installers** → **Run workflow** (istalgan branch) — yoki `v1.1.0` ko‘rinishidagi teg yuboring; teg bo‘yicha yig‘ishlar o‘rnatuvchilarni qo‘shimcha ravishda GitHub Release *qoralamasiga* biriktiradi (inson uni nashr qilmaguncha u qoralama bo‘lib qoladi). *Run workflow* ning bitta parametri bor — `include_validated_model` (standart holatda belgilangan): **klinika variantini** (3-band) yig‘ish uchun belgini olib tashlang — *Restore validated triage model* qadami o‘tkazib yuboriladi va topshiriq xulosasi buni aytadi. Teg bo‘yicha yig‘ishlar har doim modelni o‘z ichiga oladi: `v*` tegida tiklash `--require` bilan ishga tushadi, shuning uchun model yo‘qligi modelsiz o‘rnatuvchi yaratmaydi, balki yig‘ishni xato bilan to‘xtatadi.
2. *Windows installer (NSIS x64)* topshirig‘i tugashini kuting (bu uzoq davom etadi: faqat CPU uchun PyTorch 2.11.0 / torchvision 0.26.0 bilan Python bog‘liqliklari, PyInstaller, modellar, o‘rnatuvchini siqish).
3. O‘rnatuvchi qayerda paydo bo‘lishi uning ichida validatsiyadan o‘tgan model bor-yo‘qligiga bog‘liq: repozitoriy ochiq, ish jarayoni artefaktini esa GitHub ga kirgan istalgan foydalanuvchi yuklab olishi mumkin. **To‘liq variant:** faqat `v*` tegi bo‘yicha yig‘ishdan — u shu tegning GitHub Release *qoralamasiga* biriktiriladi (**Releases** → qoralama → `Sentinel Medical AI Setup 1.1.0.exe`; qoralamani faqat repozitoriyga yozish huquqi bor foydalanuvchilar ko‘radi; bu ochiq repozitoriyning nashr qilingan relizini esa istalgan kishi yuklab olishi mumkin, shuning uchun ichida to‘liq o‘rnatuvchi qolgan qoralamani nashr qilmang). Relizi allaqachon nashr qilingan tegning ish jarayonini qayta ishga tushirish (yoki shu tegda *Run workflow*) hech narsa yuklamaydi va xato bilan tugaydi: to‘liq o‘rnatuvchi faqat reliz *qoralamasiga* tushadi — uni yangi `v*` tegi bilan yig‘ing. U hech qachon ish jarayoni artefakti sifatida yuklanmaydi: branch da `include_validated_model` belgilangan holda *Run workflow* — bu mashq (yig‘ish va tutun testi bajariladi, hech narsa yuklanmaydi, topshiriq xulosasi buni aytadi). **Klinika varianti:** ishga tushirish sahifasida **Artifacts** → `sentinel-windows-x64-NO-VALIDATED-MODEL` — ichida `Sentinel Medical AI Setup 1.1.0.exe` fayli bo‘lgan zip-arxiv (electron-builder uni `desktop-app/release/` papkasiga yozadi). Artefaktlar faqat **1 kun** saqlanadi (`retention-days: 1`): o‘rnatuvchini yig‘ishdan so‘ng darhol yuklab oling va reliz hujjatlari bilan birga saqlang.
4. Validatsiyadan o‘tgan modelsiz artefakt `sentinel-windows-x64-NO-VALIDATED-MODEL` deb nomlanadi. Agar uni ataylab yig‘gan bo‘lsangiz (`include_validated_model` belgilanmagan), bu **klinika varianti**: uni faqat aniq kompyuter uchun model paketi va litsenziya bilan birga bering (5.4-band, `docs/vendor_provisioning_ru.md`). Agar branch dagi yig‘ishdan to‘liq variantni kutgan bo‘lsangiz, `MODEL_BUNDLE_TOKEN` siri yo‘q yoki modellar repozitoriysini o‘qiy olmaydi (13.2) — bunday o‘rnatish paket import qilinmaguncha `/health` → `"status": "degraded"` deb xabar beradi. `v*` tegi bo‘yicha yig‘ish bu holda xato bilan tugaydi.

### 13.2 Sirlar: validatsiyadan o‘tgan model va kod imzosi

| Sir | Vazifasi | Berilmagan bo‘lsa |
|---|---|---|
| `MODEL_BUNDLE_URL` | Ixtiyoriy zaxira varianti: ildizida `brain_triage_finetuned/` papkasi (`MANIFEST.json`, `config.json`, `preprocessor_config.json`, `model.safetensors`) bo‘lgan zip-arxivning yoki uning ko‘p qismli `manifest.json` faylining URL manzili — masalan, yopiq GitHub Release resursi (`https://api.github.com/repos/<owner>/<repo>/releases/assets/<id>`) yoki obyekt omboridagi oldindan imzolangan havola. `packaging/fetch_model_bundle.py` uni faqat modellar repozitoriysi berilmaganda o‘qiydi (masalan, yig‘ish kompyuterida); ish jarayoni modelni har doim identifikator bo‘yicha topadi, shuning uchun bu sirni **har bir model relizida o‘zgartirish endi shart emas**. Manba qanday bo‘lmasin, yuklab olingandan so‘ng har bir fayl qayta xeshlanadi va `MANIFEST.json` bilan solishtiriladi, muzlatilgan server esa tutun testida `"status": "ok"` qaytarishi shart. | CI da hech narsa o‘zgarmaydi. |
| `MODEL_BUNDLE_TOKEN` | **Model uchun yagona muhim sir:** yopiq modellar repozitoriysini o‘qish huquqiga ega token (fine-grained PAT, *Contents: read*). Repozitoriyni `MODEL_BUNDLE_REPO` repozitoriy o‘zgaruvchisi belgilaydi (berilmagan bo‘lsa — `SHAXZODBR/SIAA-models`). Shu token bilan topshiriq identifikatori yig‘ilayotgan kodda mahkamlangan (`src/inference/validated_metrics.py`) `brain-triage-<identity12>` relizini topadi va uning `manifest.json` faylini yuklab oladi — model relizida hech narsani o‘zgartirish shart emas. | Branch dagi yig‘ish: topshiriq jurnali va xulosasida baland ogohlantirish; o‘rnatuvchi validatsiyadan o‘tgan triaj modeli **bo‘lmagan holda** yig‘iladi. `v*` tegi bo‘yicha yig‘ish: xato bilan tugaydi. |
| `WIN_CSC_LINK`, `WIN_CSC_KEY_PASSWORD` | Authenticode sertifikati (`.pfx`: base64 satr yoki URL) va uning paroli; electron-builder o‘rnatuvchi va ilovaning bajariladigan faylini imzolaydi. | Imzolanmagan yig‘ilma: birinchi ishga tushirishda Windows SmartScreen *«Windows kompyuteringizni himoya qildi»* oynasini ko‘rsatadi (**More info → Run anyway**); antivirus faylni karantinga olishi mumkin. Klinikaga joriy etishdan oldin yig‘ilmani imzolang. |
| `CSC_LINK`, `CSC_KEY_PASSWORD` | DMG topshirig‘i uchun macOS Developer ID sertifikati (`.p12`). | Imzolanmagan va notarizatsiyadan o‘tmagan DMG (Gatekeeper: o‘ng tugma → Open). |

Ochiq modellar (`scripts/download_all_models.py --only brain_tumor_class,chest,chest_checkpoint,generic_processors`) ishga tushirishlar orasida keshlanadi; validatsiyadan o‘tgan model har safar qaytadan yuklab olinadi va hech qachon repozitoriyda saqlanmaydi. Shu sababli model relizi sirlarni o‘zgartirishni talab qilmaydi: modelni `packaging/publish_model_bundle.py` bilan nashr qiling, `scripts/bump_model_release.py` o‘zgarishlarini commit qilib yuboring, so‘ngra `v*` tegini yuboring. `v*` tegi bo‘yicha yig‘ish tiklashni `--require-newest` bilan ham ishga tushiradi: modellar repozitoriysida yangiroq model relizi (kodda biriktirilganidan keyin yaratilgan `brain-triage-*` relizi) bo‘lsa, yig‘ish xato bilan tugaydi — odatda bu `scripts/bump_model_release.py` commit idan oldin yuborilgan teg; o‘sha commit ni yuboring va tegni unga qo‘ying. Eski modelni ataylab qayta yig‘ish uchun `MODEL_BUNDLE_ALLOW_OLDER` repozitoriy o‘zgaruvchisiga `1` qiymatini bering (shunda yig‘ish faqat ogohlantiradi) va yig‘ishdan keyin bu o‘zgaruvchini o‘chiring. Windows yig‘ilmasini qo‘lda takrorlash uchun Windows va Python 3.11 o‘rnatilgan kompyuterda `packaging\build_backend.ps1` va `packaging\smoke_backend.ps1` ni, so‘ngra `desktop-app\` papkasida `npm run build:win` ni bajaring.
