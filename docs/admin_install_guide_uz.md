> **Taqrizchi uchun eslatma:** ushbu hujjatdagi klinik atamalarni klinikaga topshirishdan oldin amaliyotchi rentgenolog shifokor tekshirib chiqishi shart.

# SIAA — Sentinel Medical AI · O‘rnatish va ma’muriy boshqaruv qo‘llanmasi (kasalxona IT xizmati)

Versiya 1.0.0 · Hujjat tahriri 2026-09

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
| Python backend | `<o‘rnatish>\resources\backend\` — `run_server.py`, `src\` va o‘zining `venv\` papkasi | Ilova ishga tushganda ishga tushiriladi. Uning stdout/stderr chiqishi `<ma’lumotlar papkasi>\logs\backend.log` ga yoziladi. |
| Modellar | `<o‘rnatish>\resources\backend\models\` (`SENTINEL_MODELS_DIR` bilan o‘zgartirish mumkin) | `brain_triage_finetuned\` (tasdiqlangan triaj, ~330 MB, `MANIFEST.json` bilan), `hf\` (ommaviy o‘sma klassifikatori to‘plami), ixtiyoriy boshqalar. |
| Ollama + Gemma 3 | Alohida o‘rnatuvchi (ollama.com); model `ollama pull gemma3:4b` bilan yuklanadi | Kompyuter internetdan uzilishidan **oldin** o‘rnatilishi shart. |
| Ma’lumotlar papkasi | 6-bandga qarang | Birinchi ishga tushirishda yaratiladi. |

macOS da mos yo‘llar `Sentinel Medical AI.app/Contents/Resources/backend` ichida.

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

1. Barmoq izini oling. Yoki *Sozlamalar → Litsenziya* da o‘qing (ilova AI serveridan so‘raydi), yoki kompyuterda bajaring:
   ```
   cd <o‘rnatish>\resources\backend
   venv\Scripts\python.exe -m src.utils.license fingerprint
   ```
2. Barmoq izi satrini (64 ta hex belgi) va klinika nomini yetkazib beruvchiga yuboring.
3. Qaytarilgan `license.dat` ni **ma’lumotlar papkasiga** (6-band) saqlang — *Sozlamalar → Litsenziya → license.dat ni import qilish* orqali yoki faylni u yerga qo‘lda nusxalab.
4. Ilovani qayta ishga tushiring (bu serverni ham qayta ishga tushiradi). `/health` → `license.mode` qiymati `licensed` ekanini yoki *Sozlamalar → Litsenziya* da *Litsenziyalangan*, mijoz nomi va amal qilish muddati ko‘rsatilganini tekshiring.
5. Faylni qo‘lda tekshirish: `venv\Scripts\python.exe -m src.utils.license verify --license-file <yo‘l>`.

Kompyuter barmoq izi o‘zgarsa (yangi tarmoq kartasi, xost nomi o‘zgartirilgan, klonlangan VM) yoki amal qilish muddati tugasa litsenziya kuchini yo‘qotadi — sababi *Sozlamalar → Litsenziya* da va jurnalda ko‘rsatiladi. Qayta berish uchun yetkazib beruvchiga murojaat qiling; bu orada dasturiy ta’minot demo rejimiga o‘tadi, to‘xtab qolmaydi.

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
| `license.dat` | Kompyuterga bog‘langan litsenziya | Yo‘q |
| `jwt_secret.key` | Kirish tokenlarini imzolaydigan sir (rejim 0600). Uni o‘chirish hammani tizimdan chiqaradi. | Yo‘q |
| `model_key.bin` | Shifrlangan model og‘irliklari uchun har bir o‘rnatishga xos sir (reliz shifrlangan og‘irliklar bilan kelsa) | Yo‘q |

Yuklangan DICOM fayllari tahlil paytida server tanlagan nomlar bilan vaqtinchalik papkaga yoziladi va darhol o‘chiriladi; server sarlavha maydonlari va kichik oldindan ko‘rish tasvirini saqlaydi, tekshiruvning o‘zini emas.

### 6.1 Zaxira nusxa

Ma’lumotlar bazasi **shifrlanmagan holda** saqlanadi. Kompyuterni shifrlangan diskda ishlating (BitLocker / FileVault) va zaxira nusxani quyidagicha oling:

1. Ish stoli ilovasini yoping (bu serverni to‘xtatadi) yoki hech qanday tahlil bajarilmayotganiga ishonch hosil qiling.
2. **Butun ma’lumotlar papkasini** (6-band) tashqi diskka yoki NAS ga nusxalang — kamida `sentinel.db`, `sentinel.db-wal`, `sentinel.db-shm`, `doctor_corrections\`, `license.dat`, `jwt_secret.key`, `logs\`.
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

Jurnallarni yetkazib beruvchiga yuborishda faqat `sentinel_*.log` / `backend.log` ni yuboring; ma’lumotlar bazasini — hech qachon.

---

## 8. Ilova va modellarni yangilash

### 8.1 Ilovani yangilash

1. Ma’lumotlar papkasining zaxira nusxasini oling (6.1-band).
2. Yangi o‘rnatuvchini eski versiya ustidan ishga tushiring (Windows) yoki ilova paketini almashtiring (macOS). Ma’lumotlar papkasiga tegilmaydi.
3. Ilovani bir marta administrator sifatida ishga tushiring. Ma’lumotlar bazasi **o‘zi migratsiya qiladi** (yangi ustunlar qo‘shiladi; hech narsa o‘chirilmaydi). `/health` → `version` ni va kirish ekranidagi versiyani tekshiring.
4. Ish stoli ilovasi va server bir xil versiya raqamiga ega (ushbu tahrirda 1.0.0); versiya har bir PDF da chop etiladi.

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

### 8.3 Shifrlangan model og‘irliklari (reliz ular bilan kelsa)

Mahalliy o‘qitilgan detektor — yetkazib beruvchining intellektual mulki; reliz uni **diskda shifrlangan holda** (AES-256-GCM) yetkazishi mumkin. Bunday papkada `model.safetensors` o‘rniga `model.safetensors.enc` va `model.safetensors.enc.meta.json` bo‘ladi. Server og‘irliklarni yuklash paytida to‘g‘ridan-to‘g‘ri xotiraga deshifrlaydi; ochiq matn o‘rnatish diskiga hech qachon yozilmaydi.

- **Kalit** yoki ma’lumotlar papkasidagi `model_key.bin` (birinchi foydalanishda yaratiladi, rejim 0600) va kompyuter barmoq izidan hosil qilinadi, yoki yetkazib beruvchi ushbu o‘rnatish uchun bergan `SENTINEL_MODEL_KEY_HEX` orqali beriladi. `model_key.bin` yo‘qolsa yoki barmoq izi o‘zgarsa, og‘irliklarni deshifrlab bo‘lmaydi — `/health` `degraded` holatini ko‘rsatadi; yetkazib beruvchiga murojaat qiling.
- **Xeshni tekshirish** 8.2-banddagi 3-qadamdan farq qiladi: `MANIFEST.json` **ochiq matn** SHA-256 ni (`model.safetensors` yozuvi — ilova ko‘rsatadigan barmoq izi) va **shifrlangan fayl** SHA-256 ni (`model.safetensors.enc` yozuvi) qayd etadi. `certutil -hashfile model.safetensors.enc SHA256` (macOS: `shasum -a 256 model.safetensors.enc`) natijasini `.enc` yozuvi bilan, `model.safetensors.enc.meta.json` ichidagi `plaintext_sha256` ni esa `model.safetensors` yozuvi bilan solishtiring.
- Buyruq qatori vositasi (backend papkasidan): `python -m src.utils.model_crypto encrypt <model_papkasi>` (og‘irliklar → `.enc`, ochiq matn o‘chiriladi), `python -m src.utils.model_crypto decrypt <model_papkasi>` (teskari amal) va `python -m src.utils.model_crypto keyinfo` (kalit manbai va kalit barmoq izi; kalitning o‘zi emas). `encrypt`/`decrypt` — yetkazib beruvchining reliz tayyorlash qadamlari; ularni hech qachon ishlayotgan `models\` daraxtida bajarmang, faqat nusxada. Klinika kompyuterida odatda faqat `keyinfo` kerak bo‘ladi — yetkazib beruvchi so‘raganda.

---

## 9. Muhit o‘zgaruvchilari

Server jarayoni uchun beriladi (yig‘ilgan distributivda ish stoli qobig‘i `SENTINEL_REQUIRE_AUTH=1` va `SENTINEL_DATA_DIR` ni o‘zi o‘rnatadi).

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
| `SENTINEL_MODEL_KEY_HEX` | `model_key.bin` + kompyuter barmoq izidan hosil qilinadi | Shifrlangan model og‘irliklari uchun xom AES kaliti (`.enc` og‘irliklar bilan keladigan relizlar; 8.3-band) |
| `SENTINEL_MODEL_DIR_OVERRIDE_<KEY>` | berilmagan | Bitta detektorni boshqa papkaga yo‘naltirish (faqat yetkazib beruvchining A/B sinovlari) |
| `SENTINEL_DEFAULT_LANG` | `ru` | Ixtiyoriy papka kuzatuvchisi uchun hisobot tili |
| `OLLAMA_KEEP_ALIVE` (Ollama ning o‘z o‘zgaruvchisi) | — | masalan `24h` — Gemma ni xotirada ushlab turadi, kunning birinchi hisoboti sekin bo‘lmasligi uchun |

---

## 10. Nosozliklarni bartaraf etish

**`GET http://127.0.0.1:8000/health`** dan boshlang (kompyuterdagi brauzerda oching; kirish talab qilinmaydi). U quyidagini qaytaradi:

```json
{ "status": "ok" | "degraded" | "error", "version": "1.0.0", "device": "cpu",
  "auth_required": true, "offline": true,
  "models": { "brain_triage": {"loaded": true, "reason": "…"},
              "brain_tumor_class": {"loaded": true, "reason": "…"},
              "chest": {"loaded": false, "reason": "not loaded"} },
  "llm": { "backend": "ollama" | "template" | null, "reachable": true },
  "license": { "mode": "licensed" | "unlicensed" | "demo" | "dev" },
  "data_dir": "…", "uptime_seconds": 123.4 }
```

| Belgi | Qayerga qarash | Yechim |
|---|---|---|
| Kirish ekranida **AI serveri: ishlamayapti**; qizil banner *AI serveri ishlamayapti* | `/health` javob bermaydi. `<ma’lumotlar papkasi>\logs\backend.log` va eng yangi `sentinel_*.log` | `resources\backend\run_server.py` va uning `venv` papkasi mavjudligini; 8000 portni boshqa dastur band qilmaganini tekshiring; jurnaldagi oxirgi traceback ni o‘qing. Ish stoli qobig‘i serverni 3 marta qayta ishga tushiradi (2 s / 4 s / 8 s), so‘ngra jurnal yo‘li bilan dialog ko‘rsatadi. Ilovani qayta ishga tushiring; davom etsa, kompyuterni qayta yoqing; keyin jurnal bilan yetkazib beruvchiga murojaat qiling. |
| Banner *AI serveri qisman ishlamoqda: tasdiqlangan model yuklanmadi* | `/health` → `status: degraded`, `models.brain_triage.loaded: false`, `reason` ni o‘qing | Odatda **model yo‘q**: `brain_triage_finetuned` papkasi yo‘q, to‘liq emas yoki ko‘chirilgan. Uni o‘rnatish tashuvchisidan / yetkazib beruvchi paketidan tiklang va qayta ishga tushiring. Ungacha panel faqat «tasdiqlash kutilmoqda» holatidagi natijalarni beradi — rentgenolog shifokorlarga unga tayanmaslikni ayting. |
| `models.brain_tumor_class.loaded: false` | `reason` qidirilgan papkalarni ko‘rsatadi | Model to‘plamini (`models\hf\…`) o‘rnatish tashuvchisidan nusxalang. Halokatli emas: triaj detektori ishlashda davom etadi. |
| Holat panelida **LLM: mavjud emas**; bildirishnoma *Shablon hisobot — AI yordamchisi mavjud emas* | `/health` → `llm.reachable: false` | Ollama ishlamayapti yoki Gemma modeli yo‘q: `ollama list`, `ollama serve`, `ollama pull gemma3:4b`. Topilmalar va imzolashga ta’sir qilmaydi. |
| *Sozlamalar → Litsenziya* da **Litsenziyasiz** / **Demo rejim**; tahlillar «Demo limit reached (10 analyses/day)» bilan to‘xtaydi | `/license/status` → `info.reason` | `no license file at …` → `license.dat` ni ma’lumotlar papkasiga nusxalang; `invalid signature` → fayl buzilgan yoki tahrirlangan, yangi nusxa oling; `bound to a different machine` → apparat/xost nomi o‘zgargan, yangi barmoq izini yetkazib beruvchiga yuboring; `license expired on …` → uzaytiring. Faylni almashtirgandan keyin qayta ishga tushiring. |
| Shifokor imzolay olmaydi: *Sizning rolingiz hisobot imzolashga ruxsat bermaydi* | `/auth/me` dagi rol | Faqat *radiologist* va *admin* imzolay oladi; laborantlar — yo‘q. Foydalanuvchini to‘g‘ri rol bilan qayta yarating. |
| *Bu tildagi hisobot allaqachon imzolangan* (HTTP 409) | kutilgan xatti-harakat | Har bir tekshiruv va til uchun bitta imzolangan hisobot. Xato emas. |
| Tekshiruv **Ko‘rib chiqish kerak** bilan qaytadi | «Topilmalar» yorlig‘i sababini ko‘rsatadi | Bosh miyaga oid bo‘lmagan MRT yoki modeli o‘rnatilmagan modallik. Tekshiruv AI siz o‘qilishi kerak. Nosozlik emas. |
| Yuklash rad etildi: *No readable DICOM file in the upload* | — | Fayllar DICOM emas (skrinshotlar, ZIP, NIfTI). Tekshiruvni tomograf/PACS dan DICOM sifatida eksport qiling. |
| Hamma bir vaqtda tizimdan chiqib ketdi | `jwt_secret.key` o‘chirilgan yoki `SENTINEL_JWT_SECRET` o‘zgartirilgan | Bunday o‘zgarishdan keyin kutilgan holat; foydalanuvchilar shunchaki qayta kiradi. |
| Administrator paroli unutilgan | — | Tiklash interfeysi yo‘q. Serverni to‘xtating, `sentinel.db` ning zaxira nusxasini oling va tiklash tartibi uchun yetkazib beruvchiga murojaat qiling (yangi bir martalik administrator paroli faqat administrator hisob qaydnomasi mavjud bo‘lmaganda yaratiladi). |
| PDF eksporti bajarilmaydi | «Hisobot» yorlig‘idagi bildirishnoma *PDF eksport qilib bo‘lmadi* | Bo‘sh disk joyini va tanlangan papkaga yozish mumkinligini tekshiring; «Yuklab olishlar» papkasini sinab ko‘ring. |

Server uchun pytest tekshiruv testlarini (`tests\`) so‘rov bo‘yicha yetkazib beruvchi bajarishi mumkin.

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
- **Diskni shifrlang** (BitLocker / FileVault): SQLite bazasi va jurnallar ilova tomonidan shifrlanmaydi.
- Imzolangan hisobotlar sezilmas o‘zgartirishdan himoyalangan (matn SHA-256 si, imzolagan shaxs, vaqt, model identifikatori, audit yozuvi), audit jurnali esa `GET /audit/verify` bilan tekshiriladigan xesh zanjiri (7-band) — lekin baza faylining o‘zini diskka kirish huquqi bo‘lgan har kim tahrirlashi mumkin; hisob qaydnomasi va diskni himoya qiling.
- Yuklangan fayllar diskda hech qachon mijoz fayl nomlari bilan saqlanmaydi; vaqtinchalik fayllar tahlildan keyin o‘chiriladi.
- Har bir foydalanuvchiga shaxsiy hisob qaydnomasi bering, ishdan ketgan xodimlarning hisob qaydnomalarini o‘chiring va bir martalik administrator parolini umumiy hujjatlarda saqlamang.
