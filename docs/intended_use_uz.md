> **Taqrizchi uchun eslatma:** ushbu hujjatdagi klinik atamalarni klinikaga topshirishdan oldin amaliyotchi rentgenolog shifokor tekshirib chiqishi shart.

# SIAA — Sentinel Medical AI · Mo‘ljallangan foydalanish to‘g‘risidagi bayonot

Versiya 1.0.0 · Hujjat tahriri 2026-09

## Mo‘ljallangan foydalanish

Sentinel Medical AI — bosh miya MRT tekshiruvlarini tahlil qiladigan **mahalliy (on-premise) qaror qabul qilishga yordam beruvchi dasturiy ta’minot** bo‘lib, (a) patologiya bo‘lishi mumkin bo‘lgan tekshiruvlarni belgilaydi, toki ular **rentgenolog shifokor ko‘rigi uchun ustuvor tartibda** ajratilsin, va (b) rentgenolog shifokor tahrirlashi va imzolashi uchun rus, o‘zbek yoki ingliz tilida tuzilmali radiologik hisobot qoralamasini tayyorlaydi. Bu — triaj yordamchisi. U tashxis qo‘ymaydi, zararlanish joyini aniqlamaydi va hech qanday tekshiruvni norma deb e’lon qilmaydi.

## Mo‘ljallangan foydalanuvchilar

Dasturiy ta’minot o‘rnatilgan klinikaning litsenziyaga ega rentgenolog shifokorlari va vrachlari — klinikaning klinik boshqaruvi doirasida; o‘rnatish va ma’muriy boshqaruv uchun — kasalxona IT xodimlari. Bemorlar yoki tibbiy bo‘lmagan foydalanuvchilar uchun mo‘ljallanmagan.

## Ko‘rsatmalar

- **Bosh miya MRT** (kattalar uchun odatiy bosh miya protokollari; aksial T1, kontrastli T1, T2 va/yoki FLAIR seriyalari): tekshiruvlarni o‘qish uchun ustuvorlik berish. Bu — yagona tasdiqlangan ko‘rsatma.
- **Bosh KT**: dasturiy ta’minot bosh KT ni bosh suyagi ichi qon quyilishi bo‘yicha ommaviy klassifikatorga yo‘naltira oladi (agar shu model o‘rnatilgan bo‘lsa); uning natijasi *tasdiqlash kutilmoqda* deb belgilanadi va **KT uchun hech qanday aniqlik ko‘rsatkichi da’vo qilinmaydi**. Model o‘rnatilmagan bo‘lsa, tekshiruv AI natijasisiz «ko‘rib chiqish kerak» belgisi bilan qaytariladi.
- Boshqa modalliklar va tana sohalari mo‘ljallangan foydalanish doirasidan tashqarida; bunday tekshiruvlar «ko‘rib chiqish kerak» belgisi bilan qaytariladi.

## Qarshi ko‘rsatmalar va cheklovlar

- **Mustaqil tashxis, skrining yoki kasallikni istisno qilish uchun emas.** AI belgisi bo‘lmagan tekshiruv norma deb o‘qilgan *emas*: mahalliy validatsiyada har 10 ta patologik tekshiruvdan taxminan 1 tasi belgilanmagan.
- **Norma sertifikati emas.** Detektorlar paneli faqat o‘zi sanab o‘tgan topilma turlarini taniydi; boshqa har qanday patologiya u uchun ko‘rinmas.
- **Bosh miya o‘smasi klassifikatori** (glioma / meningioma / gipofiz) — mahalliy populyatsiyada validatsiyasi *kutilayotgan* ommaviy model bo‘lib, o‘smalarni ortiqcha belgilashi ma’lum. Uning belgilari — va «o‘sma yo‘q» natijasi — rentgenolog shifokorning o‘z o‘qishisiz klinik maqsadda ishlatilishi mumkin emas.
- Ishemik insult va bosh miya atrofiyasi detektorlari *eksperimental* bo‘lib, butun tekshiruv tahlilida ishga tushirilmaydi. Qon quyilishi, oq modda zararlanishi va gidrotsefaliyani aniqlash amalga oshirilmagan.
- Aniqlik quyidagilar uchun **aniqlanmagan**: bolalar, operatsiyadan yoki davolashdan keyingi bosh miya, nostandart yoki aksial bo‘lmagan protokollar, kuchli harakat yoki metall artefaktlari bo‘lgan tekshiruvlar, mahalliy o‘quv ma’lumotlarida bo‘lmagan tomograf modellari, sarlavha ma’lumotlari to‘liq bo‘lmagan DICOM eksportlari.
- Vaqt jihatidan kritik qarorlar (masalan, o‘tkir insult yoki qon quyilishini istisno qilish) uchun mo‘ljallanmagan.
- Hisobot qoralamalari, tarjimalar va *AI dan so‘rash* javoblari mahalliy til modeli yoki shablonlar tomonidan yaratiladi; ularda xatolar bo‘lishi mumkin va imzolashdan oldin so‘zma-so‘z tekshirilishi shart.
- Dasturiy ta’minot O‘zbekiston Respublikasi Sog‘liqni saqlash vazirligi yoki boshqa biror nazorat organi tomonidan tibbiy buyum sifatida sertifikatlanmagan.

## Aniqlik ko‘rsatkichlari (faqat tasdiqlangan detektor)

| Detektor | Holat | Ko‘rsatkich | Qiymat |
|---|---|---|---|
| Tekshiruv triaji — norma / patologiya (mahalliy) | tasdiqlangan | Tekshiruv darajasida sezgirlik (sensitivity) | **0,90** (131 ta patologik tekshiruvdan 118 tasi belgilangan) |
| | | Tekshiruv darajasida spetsifiklik (specificity) | **0,47** (47 ta normal tekshiruvdan 22 tasi belgilanmagan) |
| | | Aniqlik (accuracy) | 0,79 |

**Qanday o‘lchangan.** Triaj modeli (ViT-B tasvir klassifikatori) Toshkent klinikasining bosh miya MRT tekshiruvlarida qo‘shimcha o‘qitilgan — tekshiruv darajasidagi yorliqlar klinikaning o‘z radiologik hisobotlaridan olingan — va o‘qitishda ishtirok etmagan **178 ta ajratib qo‘yilgan bemorda** (131 patologik, 47 normal) sinovdan o‘tkazilgan: 890 ta aksial kesim, har bir tekshiruvga 5 ta markaziy kesim. Har bir tekshiruv uchun 5 ta kesimning patologiya ehtimollari o‘rtachalanadi va o‘rtacha qiymat ≥ 0,50 bo‘lsa tekshiruv belgilanadi — bu dasturiy ta’minot ishlab chiqarishda bajaradigan protokolning o‘zi. Baholash skripti: `scripts/eval_study_level.py`, natijalar `models/brain_triage_finetuned/study_level_eval.json` faylida (baholash sanasi 2026-09-07). Sezgirlik 0,85 dan yoki spetsifiklik 0,40 dan past bo‘lsa, xuddi shu skript model relizini bloklaydi. Normal tekshiruvlar sonining kamligiga (47) e’tibor bering: spetsifiklik bahosi noaniq (95 % ishonch oralig‘i taxminan 0,33–0,61). Yorliqlar hisobotlardan olingan va tekshiruv darajasida qo‘llangan, har bir kesim uchun ekspert belgilashi emas. Ushbu ko‘rsatkichlarni har bir o‘rnatishda tasdiqlash uchun rentgenolog shifokorlar baholagan pilot test to‘plami talab qilinadi (`pilot_evaluation_protocol_uz.md` ga qarang).

Bosh miya o‘smasi klassifikatori, KT va hisobot matni sifati uchun hech qanday aniqlik ko‘rsatkichi da’vo qilinmaydi.

## Mas’uliyat to‘g‘risidagi bayonot

Sentinel Medical AI faqat qaror qabul qilishga yordam beradi. **Malakali rentgenolog shifokor har bir tekshiruvni to‘liq o‘qiydi va har bir hisobotni imzolaydi; imzolangan hisobot — rentgenolog shifokorning kasbiy harakati va mas’uliyati.** AI natijasi hech qachon klinik qarorning yagona asosi bo‘lmasligi kerak. Rentgenolog shifokorlardan har bir yolg‘on yoki o‘tkazib yuborilgan AI belgisi gumoni haqida klinikaning pilot rahbariga xabar berish so‘raladi, toki modellarni tasdiqlash va yaxshilash mumkin bo‘lsin.

## Versiya va model identifikatsiyasi

| Element | Identifikator |
|---|---|
| Ish stoli ilovasi / AI serveri | 1.0.0 (kirish ekranida, holat panelida, `/health` da va har bir PDF da ko‘rsatiladi) |
| Tasdiqlangan triaj modeli | `models/brain_triage_finetuned` — `model.safetensors` SHA-256 `0d559766ce58…` (`MANIFEST.json` da qayd etilgan; dastlabki 12 ta belgi ilovada ko‘rsatiladi va hisobotlarda chop etiladi) |
| Bosh miya o‘smasi klassifikatori (tasdiqlash kutilmoqda) | ommaviy Hugging Face model to‘plami; identifikatori (nom + SHA-256 barmoq izi + holat) har bir natija bilan ko‘rsatiladi |
| Qaror chegarasi | 0,50 (har bir hisobotda chop etiladi) |
| Hisobot til modeli | Ollama orqali Gemma 3, mahalliy; shablonga qaytish rejimi |

Har bir natija va har bir PDF uni yaratgan modellarning identifikatorlari va tasdiqlash holatini o‘z ichiga oladi, shuning uchun har qanday hisobotni aniq model relizigacha kuzatish mumkin.
