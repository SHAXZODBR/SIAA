> **Taqrizchi uchun eslatma:** ushbu hujjatdagi klinik atamalarni klinikaga topshirishdan oldin amaliyotchi rentgenolog shifokor tekshirib chiqishi shart.

# SIAA — Sentinel Medical AI · Pilot baholash protokoli (2–3 oy)

Versiya 1.0.0 · Hujjat tahriri 2026-09

## 1. Maqsad

Ushbu klinikada va ushbu klinikaning tomograflarida AI triaj belgisi rentgenolog shifokorlar bilan qanchalik mos kelishini va dasturiy ta’minot o‘qish ish jarayoniga qanchalik mos tushishini o‘lchash — muntazam foydalanish to‘g‘risida biror qaror qabul qilinishidan oldin. Pilot **klinik yordamni o‘zgartirmaydi**: har bir tekshiruv xuddi bugungidek rentgenolog shifokor tomonidan o‘qiladi va imzolanadi; AI natijasi keyin solishtiriladi.

## 2. Qamrov

- **Qamrovga kiradi:** pilot davrida olingan va ishtirokchi rentgenolog shifokorlar tomonidan o‘qilgan kattalar bosh miya MRT tekshiruvlari.
- **Baholashning asosiy ob’ekti:** *Tekshiruv triaji — norma / patologiya (mahalliy)* detektori (holati *tasdiqlangan*).
- **Ikkilamchi (faqat tavsifiy):** *Bosh miya o‘smasi* detektori (holati *tasdiqlash kutilmoqda*), hisobot qoralamalarining foydaliligi va tizimning ishga yaroqliligi.
- **Qamrovdan tashqarida:** KT, bosh miyaga oid bo‘lmagan MRT, bolalar tekshiruvlari va AI natijasidan o‘qishni o‘tkazib yuborish yoki qisqartirish uchun har qanday foydalanish.
- **Davomiyligi:** 8–12 hafta yoki baholangan to‘plam maqsadli hajmga yetguncha (6-band) — qaysi biri kechroq bo‘lsa.

## 3. Rollar

| Rol | Vazifalar |
|---|---|
| **Pilot rahbari** (katta rentgenolog shifokor) | Protokol, baholash jurnali va yakuniy hisobot uchun javobgar; kelishmovchiliklarni hal qiladi; xavfsizlik hodisalari haqida xabar beradi. |
| **Ishtirokchi rentgenolog shifokorlar** (≥ 2) | Odatdagidek o‘qiydi va imzolaydi; AI belgilarini baholaydi; o‘qish vaqtini qayd etadi. |
| **Kasalxona IT administratori** | O‘rnatish, zaxira nusxalar, ishlash vaqti jurnali, yakunda identifikatorlarsiz eksport. |
| **Ma’lumotlar mas’uli** (pilot rahbari bo‘lishi mumkin) | Baholash jurnalini yuritadi; hech qanday shaxsiy tibbiy ma’lumot (PHI) klinikadan tashqariga chiqmasligini ta’minlaydi. |
| **Yetkazib beruvchi kontakti** | Model/ilova yangilanishlari (o‘lchov davrida — kelishilmagan bo‘lsa, yo‘q), texnik yordam; faqat jamlangan ko‘rsatkichlarni oladi. |

## 4. Ish jarayoni

1. **Boshlang‘ich davr (1–2-haftalar, ixtiyoriy, lekin tavsiya etiladi):** rentgenolog shifokorlar AI panelini ochmasdan o‘qiydi va imzolaydi; pilot rahbari solishtirish uchun ≥ 30 ta tekshiruv bo‘yicha o‘qish vaqtini qayd etadi (7.3-band).
2. **Pilot o‘qishi (har bir tekshiruv):**
   1. Butun tekshiruvni Sentinel ga yuklang (tekshiruvlar ro‘yxati → Tekshiruvni yuklash → Papka). Rentgenolog shifokor o‘qiyotgan paytda tahlil bajariladi.
   2. Rentgenolog shifokor tekshiruvni **odatdagidek** o‘qiydi, hisobotni yozadi (AI qoralamasidan boshlang‘ich nuqta sifatida foydalanish va uni erkin tahrirlash mumkin) va uni ilovada **imzolaydi**.
   3. Faqat **imzolagandan keyin** rentgenolog shifokor **Topilmalar** yorlig‘ini ochadi va har bir AI belgisini baholaydi (5-band). Imzolangan hisobot — etalon; AI uni hech qachon o‘zgartirmaydi.
3. **Har hafta:** pilot rahbari baholash jurnalini ko‘rib chiqadi, sonlarni maqsad bilan solishtiradi va har qanday xavfsizlik hodisasi yoki foydalanish qulayligi muammosini qayd etadi.
4. **Pilot yakuni:** IT identifikatorlarsiz ko‘rsatkichlarni eksport qiladi (8-band); pilot rahbari hisobot yozadi (10-band).

O‘lchov davrida model yoki ilova yangilanishi shu model versiyasi uchun hisobni qaytadan boshlaydi — aynan shuning uchun model barmoq izi har bir tekshiruv bilan qayd etiladi.

## 5. Har bir AI belgisini bir bosish bilan baholash

Har bir tahlil qilingan tekshiruv uchun rentgenolog shifokor imzolagandan keyin **har bir detektor bo‘yicha bitta qaror** beradi:

| Qaror | Ma’nosi |
|---|---|
| **Roziman** | AI belgisi (yoki «belgilanmagan») imzolangan hisobotga mos keladi: belgilangan va hisobotda ahamiyatli patologiya tasvirlangan; yoki belgilanmagan va hisobot normal / ahamiyatli topilma yo‘q. |
| **Rozi emasman** | AI belgisi imzolangan hisobotga zid: belgilangan, lekin tekshiruv normal (yolg‘on belgi), yoki belgilanmagan, lekin hisobotda ahamiyatli patologiya tasvirlangan (o‘tkazib yuborilgan). |
| **Qo‘llanilmaydi** | AI tekshiruvni tahlil qilmagan (rad etilgan / ko‘rib chiqish kerak), tekshiruv qamrovdan tashqarida yoki hisobot noaniq. Sezgirlik/spetsifiklik hisobidan chiqariladi. |

Qaror bir necha soniya oladi va erkin matn talab qilmaydi. Uni **baholash jurnaliga** (ma’lumotlar mas’uli yuritadigan elektron jadval; shablon quyida) kiriting. Tekshiruvni ilovaning «Tafsilotlar» yorlig‘ida ko‘rsatilgan **tekshiruv ID** si (server yaratgan identifikator) bo‘yicha aniqlang — bemor ismi bo‘yicha **emas**.

```
study_id | date | radiologist | modality | ai_triage_flag (yes/no) | triage_decision (agree/disagree/na) |
tumor_flag (yes/no/none) | tumor_decision (agree/disagree/na) | report_finding_class (normal/tumor/stroke/atrophy/hemorrhage/other) |
draft_used (yes/no) | time_to_read_min | model_fingerprint | comment (optional, no PHI)
```

«Ahamiyatli patologiya» ta’rifini pilot rahbari oldindan belgilaydi (tavsiya: davolash taktikasini o‘zgartiradigan yoki kuzatuvni talab qiladigan har qanday topilma; tasodifiy normal variantlar norma hisoblanadi). Qaror bo‘yicha kelishmovchiliklarni pilot rahbari hal qiladi; uning qarori yakuniy va qayd etiladi.

## 6. Baholangan test to‘plami

- **Maqsad: ≥ 100 ta baholangan bosh miya MRT tekshiruvi, shu jumladan ≥ 40 ta normal deb baholangan tekshiruv** (imzolangan hisobotda ahamiyatli topilma yo‘q). Normal tekshiruvlar — kam uchraydigan sinf va spetsifiklik bahosining aniqligini belgilaydi; har bir ketma-ket normal tekshiruvni kiriting, tanlab olmang.
- Pilot davrida **ketma-ket** tekshiruvlarni kiriting; «qiziqarli» holatlarni tanlab olmang.
- Har bir bemorga bitta tekshiruv (bemor ikki marta tekshirilgan bo‘lsa, birinchisini qoldiring).
- To‘plam klinika ichida qoladi; u ushbu va kelajakdagi model versiyalari uchun qulflangan etalon.

60 ta patologik va 40 ta normal tekshiruvda 95 % ishonch oraliqlari sezgirlik uchun taxminan ±8–10 ball, spetsifiklik uchun ±15 ballni tashkil etadi — tasdiqlangan ko‘rsatkichlardan jiddiy og‘ishni aniqlash uchun yetarli, kichik farqlarni isbotlash uchun yetarli emas.

## 7. Hisobot beriladigan ko‘rsatkichlar

### 7.1 Asosiy (triaj detektori, tekshiruv darajasida)

- **Sezgirlik** = belgilangan ÷ (barcha patologik, baholangan); **spetsifiklik** = belgilanmagan ÷ (barcha normal, baholangan); har biri **95 % ishonch oralig‘i** bilan (Wilson usuli).
- Kuzatilgan tarqalganlikdagi ijobiy va salbiy bashorat qiymati (PPV, NPV); umumiy belgilash darajasi.
- Yetkazib beruvchining tasdiqlangan ko‘rsatkichlari (178 ta mahalliy bemorda sezgirlik 0,90, spetsifiklik 0,47) va reliz darvozasi pollari (0,85 / 0,40) bilan solishtirish: pilot ishonch oralig‘ining **quyi chegarasi** har bir poldan yuqori ekanini ko‘rsating.

### 7.2 Ikkilamchi (tavsifiy)

- O‘sma detektori: roziman / rozi emasman sonlari; normal tekshiruvlar orasida yolg‘on belgilar ulushi.
- «Ko‘rib chiqish kerak» belgisi bilan qaytgan tekshiruvlar ulushi va sabablari; tahlil xatolari.
- Hisobot qoralamalari: qoralama boshlang‘ich nuqta sifatida ishlatilgan hisobotlar %; tahrirlangan % (serverning tuzatishlar jadvalidan).

### 7.3 O‘qish vaqti

Tekshiruvni ochishdan hisobotni imzolashgacha bo‘lgan daqiqalar, har bir tekshiruv uchun qayd etiladi (o‘qish jurnalidan yoki ilovaning vaqt tamg‘alaridan: tahlil vaqti va `signed_at`). Boshlang‘ich haftalar va pilot haftalari uchun mediana va kvartillararo oraliqni keltiring.

### 7.4 Tizim

Ish vaqtida ishlash vaqti (IT jurnali va `/health` tekshiruvlaridan), qayta ishga tushirishlar soni, har bir tekshiruv uchun tahlil vaqtining medianasi.

### 7.5 Xavfsizlik

AI natijasi o‘qishga noto‘g‘ri yo‘nalishda ta’sir qilishi mumkin bo‘lgan har qanday hodisa (masalan, «belgilanmagan» ni ko‘rgandan keyin haqiqiy topilma ahamiyatining pasaytirilishi). Maqsad: nol; har bir bunday hodisa natijasidan qat’i nazar hisobotda tasvirlanadi.

## 8. Ma’lumotlar bilan ishlash

- Barcha tekshiruvlar, hisobotlar, jurnallar va baholash jurnali **klinika hududida** qoladi. Ilovaning o‘zi hech qachon ma’lumot uzatmaydi.
- Baholash jurnalida tekshiruv ID lari, sanalar, qarorlar va vaqtlar bo‘ladi — **ismlar, bemor ID lari yoki tug‘ilgan sanalar bo‘lmaydi**. Uni klinika tarmog‘ida, faqat pilot jamoasi kirishi mumkin bo‘lgan joyda saqlang.
- Yakunda IT **yetkazib beruvchi uchun identifikatorlarsiz eksport** tayyorlaydi: jamlangan ko‘rsatkichlar (7-band) va, klinikaning ma’lumotlar to‘g‘risidagi kelishuvi ruxsat bersa, faqat tekshiruv ID lari bilan baholash jurnali jadvali. Tasvirlar yo‘q, hisobotlar yo‘q, ma’lumotlar bazasi fayli yo‘q.
- Baholash jurnali va ko‘rsatkichlar hisobotini klinika arxivi uchun pilot hujjatlari bilan birga saqlang; imzolangan hisobotlar odatdagidek ilovaning ma’lumotlar bazasida qoladi.
- Yetkazib beruvchi jamlangan natijalardan modellarni qayta tasdiqlash yoki qayta o‘qitish uchun faqat amaldagi klinika ma’lumotlar kelishuvi doirasida foydalanishi mumkin.

## 9. Muvaffaqiyat mezonlari

Quyidagilarning **barchasi** bajarilganda pilot muvaffaqiyatli hisoblanadi va muntazam foydalanish ko‘rib chiqilishi mumkin:

1. ≥ 100 ta baholangan tekshiruv, shu jumladan ≥ 40 ta normal.
2. Triaj sezgirligining nuqtaviy bahosi ≥ 0,85 **va** spetsifiklik ≥ 0,40, ishonch oralig‘i ko‘rsatilgan holda; quyi chegaralar ustuvorlik berish uchun maqbulmi — pilot rahbari hal qiladi.
3. AI natijasi bilan bog‘liq xavfsizlik hodisasi yo‘q.
4. Ish vaqtida tizimning ishlash vaqti ≥ 95 %; tahlil kutish vaqtining medianasi rentgenolog shifokorlar uchun maqbul.
5. O‘qish vaqti boshlang‘ich davrdan yomon emas (mediana).
6. Rentgenolog shifokorlar bahosi (yakunda qisqa so‘rovnoma): belgi va qoralama foydali yoki neytral, zararli emas.

Agar 2-mezon bajarilmasa, model qayta tasdiqlanmaguncha ushbu klinikada ustuvorlik berish uchun ishlatilmasligi kerak; baholangan to‘plam aynan yetkazib beruvchiga uni tuzatish imkonini beradi.

## 10. Hisobot shabloni

```
PILOT BAHOLASH HISOBOTI — Sentinel Medical AI
Klinika: ____________   Davr: ______ – ______   Pilot rahbari: ____________
Dasturiy ta’minot versiyasi: 1.0.0   Triaj modeli barmoq izi: ____________   O‘sma modeli barmoq izi: ____________

1. Kiritish: tahlil qilingan tekshiruvlar ___ ; baholangan ___ (normal ___ / patologik ___); qo‘llanilmaydi ___ (sabablari)
2. Triaj detektori: sezgirlik ___ (95 % IO ___–___); spetsifiklik ___ (95 % IO ___–___); PPV ___; NPV ___; belgilash darajasi ___
   Tasdiqlangan ko‘rsatkichlar / darvoza pollari bilan solishtirish: ____________
3. O‘sma detektori (tavsifiy): roziman ___ / rozi emasman ___ ; normal tekshiruvlarda yolg‘on belgilar ___
4. Hisobot qoralamalari: ishlatilgan ___% ; tahrirlangan ___% ; odatiy tuzatishlar: ____________
5. O‘qish vaqti: boshlang‘ich mediana ___ daq. (IQR ___) ; pilot medianasi ___ daq. (IQR ___)
6. Tizim: ishlash vaqti ___% ; qayta ishga tushirishlar ___ ; tahlil vaqti medianasi ___ s ; hodisalar: ____________
7. Xavfsizlik hodisalari: ____________ (yo‘q / tasvirlangan)
8. Rentgenolog shifokorlar fikr-mulohazasi xulosasi: ____________
9. Muvaffaqiyat mezonlari bo‘yicha xulosa (1–6): har bir mezon bo‘yicha o‘tdi / o‘tmadi
10. Tavsiya: muntazam foydalanishni davom ettirish / pilotni uzaytirish / model yangilanishigacha to‘xtatish
Ilovalar: identifikatorlarsiz baholash jurnali (faqat tekshiruv ID lari), IT ishlash vaqti jurnali
Imzolar: pilot rahbari ______ ; bo‘lim mudiri ______ ; IT ______
```
