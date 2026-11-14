> **Taqrizchi uchun eslatma:** ushbu hujjatdagi klinik atamalarni klinikaga topshirishdan oldin amaliyotchi rentgenolog shifokor tekshirib chiqishi shart.

# SIAA — Sentinel Medical AI · Mo‘ljallangan foydalanish to‘g‘risidagi bayonot

Versiya 1.1.0 · Hujjat tahriri 2026-11

## Mo‘ljallangan foydalanish

Sentinel Medical AI — bosh miya MRT tekshiruvlarini tahlil qiladigan **mahalliy (on-premise) qaror qabul qilishga yordam beruvchi dasturiy ta’minot** bo‘lib, (a) patologiya bo‘lishi mumkin bo‘lgan tekshiruvlarni belgilaydi, toki ular **rentgenolog shifokor ko‘rigi uchun ustuvor tartibda** ajratilsin, va (b) rentgenolog shifokor tahrirlashi va imzolashi uchun rus, o‘zbek yoki ingliz tilida tuzilmali radiologik hisobot qoralamasini tayyorlaydi. Bu — triaj yordamchisi. U tashxis qo‘ymaydi, zararlanish joyini aniqlamaydi va hech qanday tekshiruvni norma deb e’lon qilmaydi.

## Mo‘ljallangan foydalanuvchilar

Dasturiy ta’minot o‘rnatilgan klinikaning litsenziyaga ega rentgenolog shifokorlari va vrachlari — klinikaning klinik boshqaruvi doirasida; o‘rnatish va ma’muriy boshqaruv uchun — kasalxona IT xodimlari. Bemorlar yoki tibbiy bo‘lmagan foydalanuvchilar uchun mo‘ljallanmagan.

## Ko‘rsatmalar

- **Bosh miya MRT** (kattalar uchun odatiy bosh miya protokollari; aksial T1, kontrastli T1, T2 va/yoki FLAIR seriyalari): tekshiruvlarni o‘qish uchun ustuvorlik berish. Bu — yagona tasdiqlangan ko‘rsatma.
- **Bosh KT**: KT uchun tasdiqlangan detektor yo‘q. Standart holatda (pending detektorlar o‘chiq) bosh KT AI natijasisiz, tasdiqlangan detektor yo‘qligi haqidagi izoh bilan «ko‘rib chiqish kerak» belgisi bilan qaytariladi. Faqat administrator pending detektorlarni yoqqan va model o‘rnatilgan bo‘lsa, tekshiruv bosh suyagi ichi qon quyilishi bo‘yicha ommaviy klassifikatorga yo‘naltiriladi va uning natijasi *tasdiqlash kutilmoqda* deb belgilanadi; **KT uchun hech qanday aniqlik ko‘rsatkichi da’vo qilinmaydi**.
- Boshqa modalliklar va tana sohalari mo‘ljallangan foydalanish doirasidan tashqarida; bunday tekshiruvlar «ko‘rib chiqish kerak» belgisi bilan qaytariladi.

## Qarshi ko‘rsatmalar va cheklovlar

- **Mustaqil tashxis, skrining yoki kasallikni istisno qilish uchun emas.** AI belgisi bo‘lmagan tekshiruv norma deb o‘qilgan *emas*: mahalliy validatsiyada har 10 ta patologik tekshiruvdan taxminan 1 tasi belgilanmagan.
- **Norma sertifikati emas.** Detektorlar paneli faqat o‘zi sanab o‘tgan topilma turlarini taniydi; boshqa har qanday patologiya u uchun ko‘rinmas.
- **Bosh miya o‘smasi klassifikatori** (glioma / meningioma / gipofiz) — mahalliy populyatsiyada validatsiyasi *kutilayotgan* ommaviy model bo‘lib, o‘smalarni ortiqcha belgilashi ma’lum. Uning belgilari — va «o‘sma yo‘q» natijasi — rentgenolog shifokorning o‘z o‘qishisiz klinik maqsadda ishlatilishi mumkin emas.
- **Tasdiqlanmagan detektorlar (pending va experimental) standart holatda o‘chirilgan; pending detektorlarni faqat administrator yoqa oladi va har bir bunday o‘zgarish audit jurnaliga yoziladi.**
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

**Qanday o‘lchangan.** Triaj modeli (ViT-B tasvir klassifikatori) Toshkent klinikasining bosh miya MRT tekshiruvlarida qo‘shimcha o‘qitilgan — tekshiruv darajasidagi yorliqlar klinikaning o‘z radiologik hisobotlaridan olingan — va o‘qitishda ishtirok etmagan **178 ta ajratib qo‘yilgan bemorda** (131 patologik, 47 normal) sinovdan o‘tkazilgan: 890 ta aksial kesim, har bir tekshiruvga 5 ta markaziy kesim. Har bir tekshiruv uchun 5 ta kesimning patologiya ehtimollari o‘rtachalanadi va o‘rtacha qiymat ≥ 0,50 bo‘lsa tekshiruv belgilanadi — bu dasturiy ta’minot ishlab chiqarishda bajaradigan protokolning o‘zi. Baholash skripti: `scripts/eval_study_level.py`, natijalar `models/brain_triage_finetuned/study_level_eval.json` faylida (baholash sanasi 2026-11-09). Sezgirlik 0,85 dan yoki spetsifiklik 0,40 dan past bo‘lsa, xuddi shu skript model relizini bloklaydi. Normal tekshiruvlar sonining kamligiga (47) e’tibor bering: spetsifiklik bahosi noaniq (95 % ishonch oralig‘i taxminan 0,33–0,61). Yorliqlar hisobotlardan olingan va tekshiruv darajasida qo‘llangan, har bir kesim uchun ekspert belgilashi emas. Ushbu ko‘rsatkichlarni har bir o‘rnatishda tasdiqlash uchun rentgenolog shifokorlar baholagan pilot test to‘plami talab qilinadi (`pilot_evaluation_protocol_uz.md` ga qarang).

**Yangi bemorlarda tekshiruv (2026-yil noyabr).** Ushbu sinovdan so‘ng o‘sha model o‘sha protokol bo‘yicha shu klinikaning yana 518 ta bosh miya MRT tekshiruvida (512 bemor) baholandi; ularning xulosalari keyinroq kelgan va ular na o‘qitishda, na yuqoridagi sinovda ishlatilgan: sezgirlik 0,91 (378 ta patologik tekshiruvdan 345 tasi belgilangan; 95% ishonch oralig‘i 0,88–0,94), spetsifiklik 0,34 (140 ta normal tekshiruvdan 47 tasi belgilanmagan; 95% ishonch oralig‘i 0,26–0,42). Barcha tekshiruvlarning taxminan 85% i belgilangan, belgilanmagan har 10 ta tekshiruvdan taxminan 4 tasi patologik bo‘lgan. Aniqlash topilma turiga bog‘liq: atrofiya va insult normal tekshiruvlarga qaraganda ancha ko‘p belgilanadi, mayda o‘choqli topilmalar (oq modda o‘choqlari, kistalar) esa normal tekshiruvlar bilan deyarli bir xil chastotada belgilanadi — model ularni ishonchli aniqlamaydi. Skript: `scripts/eval_prospective.py`.

**Belgi bemor yoshiga bog‘liq.** Ushbu tekshiruvda «norma» xulosalari yoshroq bemorlarga tegishli edi (yosh medianasi 29; 60 va undan katta yoshdagi birorta bemorda «norma» xulosasi bo‘lmagan), patologiyali xulosalarning aksariyati esa yoshga xos tomir yoki atrofik o‘zgarishlarni tasvirlagan. Bemor yoshining o‘zi patologik va normal tekshiruvlarni model bilan bir xil darajada ajratdi (ikkalasida AUC 0,78). 40 yoshgacha sezgirlik 0,78 (132 tadan 103 tasi), spetsifiklik 0,42 (110 tadan 46 tasi) bo‘ldi; 40 yoshdan boshlab belgi xulosadan qat’i nazar deyarli har bir tekshiruvga qo‘yildi (30 ta normal tekshiruvdan 29 tasi belgilangan). Yoshi yaqin (farqi 5 yilgacha) bemorlarning «patologiya — norma» juftliklarida model patologik tekshiruvni 62% juftlikda yuqoriroq baholadi. Shuning uchun 40 va undan katta yoshdagi bemorda belgining o‘zi deyarli ma’lumot bermaydi. Model tez o‘qilishi kerak bo‘lgan topilmalarni ajratib ko‘rsatishi ham isbotlanmagan: xulosasida shunday topilma (hajmli hosila, yaqinda o‘tkazilgan infarkt, qon quyilishi, demiyelinizatsiya; xulosa matni bo‘yicha dastlabki ta’rif) ko‘rsatilgan 46 ta tekshiruvdan 42 tasi belgilangan, biroq qolgan 472 tadan 396 tasi ham belgilangan.

Bosh miya o‘smasi klassifikatori, KT va hisobot matni sifati uchun hech qanday aniqlik ko‘rsatkichi da’vo qilinmaydi.

## Mas’uliyat to‘g‘risidagi bayonot

Sentinel Medical AI faqat qaror qabul qilishga yordam beradi. **Malakali rentgenolog shifokor har bir tekshiruvni to‘liq o‘qiydi va har bir hisobotni imzolaydi; imzolangan hisobot — rentgenolog shifokorning kasbiy harakati va mas’uliyati.** AI natijasi hech qachon klinik qarorning yagona asosi bo‘lmasligi kerak. Rentgenolog shifokorlardan har bir yolg‘on yoki o‘tkazib yuborilgan AI belgisi gumoni haqida klinikaning pilot rahbariga xabar berish so‘raladi, toki modellarni tasdiqlash va yaxshilash mumkin bo‘lsin.

## Versiya va model identifikatsiyasi

| Element | Identifikator |
|---|---|
| Ish stoli ilovasi / AI serveri | 1.1.0 (kirish ekranida, holat panelida, `/health` da va har bir PDF da ko‘rsatiladi) |
| Tasdiqlangan triaj modeli | `models/brain_triage_finetuned` — `model.safetensors` SHA-256 `0d559766ce58…` (`MANIFEST.json` da qayd etilgan; dastlabki 12 ta belgi ilovada ko‘rsatiladi va hisobotlarda chop etiladi) |
| Bosh miya o‘smasi klassifikatori (tasdiqlash kutilmoqda) | ommaviy Hugging Face model to‘plami; identifikatori (nom + SHA-256 barmoq izi + holat) har bir natija bilan ko‘rsatiladi |
| Qaror chegarasi | 0,50 (har bir hisobotda chop etiladi) |
| Hisobot til modeli | Ollama orqali Gemma 3, mahalliy; shablonga qaytish rejimi |

Har bir natija va har bir PDF uni yaratgan modellarning identifikatorlari va tasdiqlash holatini o‘z ichiga oladi, shuning uchun har qanday hisobotni aniq model relizigacha kuzatish mumkin.
