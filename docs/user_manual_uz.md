> **Taqrizchi uchun eslatma:** ushbu hujjatdagi klinik atamalarni klinikaga topshirishdan oldin amaliyotchi rentgenolog shifokor tekshirib chiqishi shart.

# SIAA — Sentinel Medical AI · Rentgenolog shifokorlar uchun foydalanuvchi qo‘llanmasi

Versiya 1.0.0 (ish stoli ilovasi va AI serveri) · Hujjat tahriri 2026-09

---

## 1. Bu mahsulot nima — va nima emas

**Sentinel Medical AI — qaror qabul qilishga yordam beruvchi vosita.** U klinika ichidagi kompyuterda ishlaydi, bosh miya MRT tekshiruvini o‘qiydi, diqqat bilan ko‘rib chiqish talab qilishi mumkin bo‘lgan tekshiruvlarni belgilaydi va rus, o‘zbek yoki ingliz tilida tuzilmali hisobot (xulosa) qoralamasini tayyorlaydi.

U **diagnostika vositasi emas** va **rentgenolog shifokorning o‘rnini bosmaydi**:

- **Har bir tekshiruvni rentgenolog shifokor o‘qiydi va imzolaydi.** AI natijasi — faqat maslahat. Imzolangan hisobot AI niki emas, rentgenolog shifokorniki.
- **AI hech qachon tekshiruvni norma deb tasdiqlamaydi.** Hech qanday topilma belgilanmaganda ekranda *«AI hech qanday topilma belgilamadi — bu norma xulosasi EMAS»* yozuvi chiqadi. AI faqat «Topilmalar» yorlig‘ida sanab o‘tilgan topilma turlarini tekshiradi; qolgan hamma narsa u uchun ko‘rinmas.
- **Mahalliy bemorlarda faqat bitta detektor tasdiqlangan** (*tekshiruv triaji* detektori). *Bosh miya o‘smasi* detektori — ommaviy model bo‘lib, ushbu klinika populyatsiyasida uni tasdiqlash hali kutilmoqda; u o‘smalarni ortiqcha belgilashi ma’lum.
- Dasturiy ta’minot O‘zbekiston Respublikasi Sog‘liqni saqlash vazirligi yoki boshqa biror nazorat organi tomonidan tibbiy buyum sifatida sertifikatlanmagan. Undan klinikaning o‘z klinik boshqaruvi doirasida foydalaning.

Hamma narsa klinika kompyuterida, internetsiz ishlaydi. Hech qanday tasvir, hisobot yoki bemor identifikatori kompyuterdan tashqariga chiqmaydi.

---

## 2. Tizimga kirish va parolni o‘zgartirish

Hisob qaydnomalarini administrator yaratadi. Rollar: **rentgenolog shifokor** (tahlil qilish, hisobotni tahrirlash va imzolash), **administrator** (shu bilan birga foydalanuvchilarni boshqarish va audit jurnali), **rentgen laboranti** (faqat yuklash; tahlil qila olmaydi va imzolay olmaydi).

1. *Sentinel Medical AI* ni «Пуск» menyusidan (Windows) yoki «Applications» papkasidan (macOS) ishga tushiring. AI serveri fon rejimida ishga tushadi; kirish ekranining pastidagi holat qatorida **AI serveri: ishlamoqda** yozuvi chiqishini kuting.
2. Interfeys tilini tanlang (РУС / O‘ZB / ENG) — bu yangi hisobotlarning standart tili ham bo‘ladi.
3. Foydalanuvchi nomi va parolni kiriting va **Kirish** tugmasini bosing.
   [Skrinshot: til almashtirgichi va AI serveri holat qatori bilan kirish ekrani]
4. **Vaqtinchalik parol bilan birinchi kirish:** ilova boshqa hamma narsadan oldin *Parolni o‘zgartirish* shaklini ochadi. Vaqtinchalik parolni, so‘ngra yangi parolni (kamida 8 ta belgi, ikki marta) kiriting. Buni bajarmaguningizcha ish stantsiyasi qulflangan holda qoladi.
5. Sessiya 12 soat davom etadi. Muddati tugagach, siz kirish ekraniga qaytarilasiz (*Sessiya tugadi — qayta kiring*). Imzolanmagan yoki eksport qilinmagan hamma narsa serverda saqlanib qoladi va kirganingizdan keyin tekshiruvlar ro‘yxatida yana paydo bo‘ladi.

Parolni keyinroq o‘zgartirish uchun administratoringizga murojaat qiling (server sizning hisobingizni *parolni o‘zgartirish shart* deb belgilaganida ilova parolni o‘zgartirish shaklini o‘zi ko‘rsatadi).

Tizimdan chiqish: **Fayl → Chiqish**. Chiqishda ekrandagi barcha bemor ob’ektlari tozalanadi.

---

## 3. Dastlabki sozlash (har bir ish stantsiyasi uchun bir marta)

O‘rnatishdan keyingi birinchi ishga tushirishda ilova besh qadamli **Dastlabki sozlash** ustasini ochadi: *Til → Klinika → AI serveri → Litsenziya → Qo‘llab-quvvatlash*. Odatda uni IT administratoringiz bajaradi; agar u sizga chiqsa, IT xizmatiga murojaat qiling. Ustani keyinroq **Sozlamalar → Klinika → Sozlashni qayta boshlash** orqali qayta ochish mumkin. U yerda kiritilgan klinika nomi har bir PDF sarlavhasida chop etiladi.

---

## 4. Tekshiruvlar ro‘yxati

Chap panel — **tekshiruvlar ro‘yxati**: ushbu AI serveriga ma’lum bo‘lgan tekshiruvlar, eng yangilari yuqorida.

[Skrinshot: qidiruv maydoni, modallik tugmalari va holat belgilari bilan tekshiruvlar ro‘yxati]

- **Qidiruv** — bemor ID, yo‘llanma raqami yoki modallik bo‘yicha. **Filtr** — qidiruv maydoni ostidagi modallik tugmalari orqali. **Saralash** — vaqt, bemor, modallik yoki holat bo‘yicha.
- Har bir qatorda bemor ID, modallik va soha, tekshiruv sanasi hamda holat ko‘rsatiladi: **Kutmoqda → Tahlil → Yakunlangan** yoki **Xato**. **Ko‘rib chiqish kerak** belgisi AI tekshiruvni halol tahlil qila olmaganini bildiradi (5.3-bandga qarang) — uni rentgenolog shifokor to‘liq o‘qishi shart.
- Kirganingizdan keyin tekshiruvlar ro‘yxati **serverdan tiklanadi** (oxirgi 100 ta tekshiruv). Tiklangan tekshiruvni tanlasangiz, uning AI natijasi, hisobot qoralamalari va imzolangan hisobotlari qayta yuklanadi.
- Tekshiruvni ko‘ruvchida va o‘ng panelda ochish uchun qatorni bosing.

---

## 5. Tekshiruvni yuklash

### 5.1 Qadamlar

1. Tekshiruvlar ro‘yxatida **Tekshiruvni yuklash**, so‘ngra **Papka** (tavsiya etiladi — butun tekshiruv, barcha seriyalar) yoki **Fayllar** tugmasini bosing. Papka yoki fayllarni tekshiruvlar ro‘yxatining pastki qismiga **sudrab tashlash** ham mumkin.
   [Skrinshot: «Tekshiruvni yuklash» tugmalari va tashlash maydoni]
2. *Fayllar o‘qilmoqda…*, so‘ngra *N ta fayl yuklanmoqda · x%*, so‘ngra *N ta fayl tahlil qilinmoqda…* yozuvlarini kuting. Tekshiruv ro‘yxatda **Tahlil** holatida paydo bo‘ladi va natija kelganda **Yakunlangan** holatiga o‘tadi.
3. Ko‘ruvchi **model aslida tahlil qilgan kesimni** ko‘rsatadi, o‘ng panel esa **Topilmalar** yorlig‘ida ochiladi.

Agar holat qatorida *AI serveri: ishlamayapti* yozuvi bo‘lsa, yuklash tugmalari o‘chirilgan bo‘ladi (10-bandga qarang).

### 5.2 Qabul qilinadigan formatlar

- **DICOM** fayllari: `.dcm`, `.dicom` yoki kengaytmasiz fayllar (tomograflarning odatiy eksporti). Yashirin fayllar va `DICOMDIR` avtomatik o‘tkazib yuboriladi. Papka yuklanganda ichki papkalar ham kiritiladi.
- Siqilgan DICOM (JPEG 2000, JPEG-LS, RLE) va Philips, GE, Siemens tomograflarining ko‘p kadrli (multi-frame) fayllari qo‘llab-quvvatlanadi.
- Qabul **qilinmaydi**: NIfTI, JPEG/PNG skrinshotlar, PDF, ZIP arxivlar (avval arxivdan chiqaring). DICOM fayllari bo‘lmagan tanlov *Tanlangan fayllar orasida DICOM (.dcm) yo‘q* xabari bilan rad etiladi.
- **Butun bosh miya MRT tekshiruvini** yuklang. Panel kerakli aksial seriyalarni o‘zi tanlaydi (kontrastli T1, T1, T2, FLAIR); faqat bitta seriya bo‘lsa ham ishlaydi — o‘sha seriya bo‘yicha.

### 5.3 AI tahlil qilmaydigan tekshiruvlar

- **Bosh miyaga oid bo‘lmagan MRT** (umurtqa, tizza, qorin bo‘shlig‘i …) bosh miya paneli tomonidan rad etiladi: tekshiruv **Ko‘rib chiqish kerak** belgisi va *Tahlil qilinmadi — rentgenolog to‘liq ko‘rib chiqishi kerak* yozuvi bilan saqlanadi. Uni odatdagidek o‘qing.
- **Boshqa modalliklar** (KT, rentgen, mammografiya) boshqa detektorlarga faqat serveringizda mos model o‘rnatilgan bo‘lsa yo‘naltiriladi; aks holda ular ham **Ko‘rib chiqish kerak** belgisi bilan qaytadi. Tasdiqlangan triaj detektori **faqat bosh miya MRT** ni qamrab oladi.
- DICOM sifatida o‘qib bo‘lmaydigan tekshiruv xato xabari bilan rad etiladi.

---

## 6. «Topilmalar» yorlig‘ini o‘qish

[Skrinshot: ogohlantirish banneri, umumiy baho kartasi, ikki belgili topilma kartalari va modellar ro‘yxati bilan «Topilmalar» yorlig‘i]

### 6.1 Ogohlantirish banneri

Har bir natija bir xil eslatma bilan boshlanadi: *«AI triaj yordamchisi. U faqat sanab o‘tilgan topilma turlarini belgilaydi va tekshiruvni norma deb TASDIQLAY OLMAYDI. „Tasdiqlash kutilmoqda“ deb belgilangan topilmalar ushbu klinika populyatsiyasida tasdiqlanmagan; „eksperimental“ topilmalar — faqat skrining ishoralari. Har bir tekshiruvni rentgenolog shifokor o‘qiydi va imzolaydi.»* U har bir PDF da ham chop etiladi.

### 6.2 Umumiy baho

- **AI topilma belgiladi — ko‘rib chiqish kerak** (qizil): kamida bitta detektor ishga tushdi. Tekshiruvni o‘qishda ustuvor tartibda ko‘rib chiqish lozim.
- **AI hech qanday topilma belgilamadi — bu norma xulosasi EMAS** (kulrang): hech bir detektor ishga tushmadi. Bu norma natijasi *emas* — tekshiruvni to‘liq o‘qing.

Karta ostida: belgilangan topilmalar soni, tahlil vaqti va qaror **chegarasi** (0,50).

### 6.3 Ushbu yig‘ilmadagi ikkita bosh miya detektori

| Detektor | Holat belgisi | Nima qiladi | Ma’lum cheklovlar |
|---|---|---|---|
| **Tekshiruv triaji — norma / patologiya (mahalliy)** | **Tasdiqlangan** | Toshkent klinikasi tekshiruvlarida o‘qitilgan va sinovdan o‘tkazilgan. Patologiya ehtimolini 5 ta markaziy aksial kesim bo‘yicha o‘rtachalaydi; o‘rtacha qiymat ≥ 0,50 bo‘lsa tekshiruvni belgilaydi. | 178 ta ajratib qo‘yilgan mahalliy bemorda tekshiruv darajasida sezgirlik (sensitivity) 0,90 va spetsifiklik (specificity) 0,47. Boshqacha aytganda: har 10 ta patologik tekshiruvdan taxminan 1 tasi **belgilanmaydi**, normal tekshiruvlarning esa taxminan yarmi **belgilanadi**. U ustuvorlik beradi; tashxis qo‘ymaydi, joylashuvni aniqlamaydi va patologiyani istisno qilmaydi. |
| **Bosh miya o‘smasi (glioma / meningioma / gipofiz)** | **Tasdiqlash kutilmoqda** | Kaggle bosh miya o‘smalari to‘plamida o‘qitilgan ommaviy 2D klassifikator; bitta markaziy kesim bo‘yicha ishlaydi. | Mahalliy tomograflarda **o‘smalarni ortiqcha belgilaydi**: «Glioma / meningioma gumoni» belgilarining ko‘pchiligi yolg‘on ijobiy. Uning «O‘sma aniqlanmadi» qatori ham xuddi shunday tekshirilmagan. Har qanday o‘sma belgisini ko‘rib chiqishga da’vat deb qabul qiling, topilma deb emas. |

**Eksperimental** deb belgilangan detektorlar (DWI bo‘yicha ishemik insult, bosh miya atrofiyasi) dasturda mavjud, lekin ushbu yig‘ilmada butun tekshiruv tahlilida **ishga tushirilmaydi**. Qon quyilishi, oq modda zararlanishi va gidrotsefaliyani aniqlash amalga oshirilmagan.

### 6.4 Har bir topilma kartasidagi belgilar

Har bir kartada **ikkita belgi** bor:

1. **Shoshilinchlik** — detektor ishga tushgan-tushmaganidan va uning tasdiqlash holatidan kelib chiqadi, ishonchlilik raqamidan hech qachon emas:
   - **KO‘RIB CHIQISH KERAK** — *tasdiqlangan* detektor tekshiruvni belgiladi.
   - **TASDIQLANMAGAN BELGI** — *tasdiqlash kutilayotgan* yoki *eksperimental* detektor nimanidir belgiladi.
   - **BELGILANMAGAN** — detektor ishga tushmadi.
2. Detektorning **tasdiqlash holati** — **Tasdiqlangan** (yashil), **Tasdiqlash kutilmoqda** (sariq), **Eksperimental** (kulrang).

Kartada shuningdek detektor, ishlatilgan MRT ketma-ketligi, xom sinf va ishonchlilik ko‘rsatiladi. *Tasdiqlash kutilayotgan* detektorning, aytaylik, 0,92 ishonchliligi baribir tasdiqlanmagan belgi hisoblanadi.

### 6.5 Modellar ro‘yxati

Pastda: ishga tushgan har bir modelning ko‘rsatiladigan nomi, 12 belgili barmoq izi (SHA-256) va holati, shuningdek AI serveri versiyasi. Xuddi shu identifikator har bir PDF da chop etiladi, shuning uchun har qanday hisobotni aniq model relizigacha kuzatish mumkin.

Bosh miya paneli zararlanish issiqlik xaritalarini yoki joylashuvini chizmaydi; ko‘ruvchi faqat tahlil qilingan kesimni ko‘rsatadi.

---

## 7. «Hisobot» yorlig‘i

[Skrinshot: РУС / O‘ZB / ENG tugmalari, tahrirlash qalami, matn maydoni va «Imzolash» / «Eksport» tugmalari bilan «Hisobot» yorlig‘i]

1. Tahlil tugagach, AI topilmalari asosida interfeys tilida **hisobot qoralamasi** yaratiladi (mahalliy til modeli mavjud bo‘lsa — u orqali; aks holda qat’iy shablon bo‘yicha — *Shablon hisobot — AI yordamchisi mavjud emas* bildirishnomasi qaysi biri ekanini aytadi).
2. **РУС / O‘ZB / ENG til tugmalari** hisobotni boshqa tilda ochadi. Asl tildan boshqa tilda ochilgan versiya imzolanmaguncha *AI avtomatik tarjimasi — tekshirish kerak* deb belgilanadi. Tugmadagi nuqta qoralama borligini, belgi (✓) esa shu tilda imzolanganini bildiradi.
3. **Tahrirlash** (qalam belgisi) matnni ochadi. Erkin tahrirlang: qoralama — faqat boshlang‘ich nuqta. *Qoralama shifokor tomonidan o‘zgartirilgan* yorlig‘i paydo bo‘ladi va tahriringiz serverda tuzatish sifatida saqlanadi (kelajakdagi modellarni yaxshilash uchun ishlatiladi; bemor identifikatorlari biriktirilmaydi).
4. Imzo chizig‘idan yuqoridagi hamma narsa siz imzolamaguningizcha **AI QORALAMASI** hisoblanadi. Qoralama PDF larida **AI QORALAMASI — IMZOLANMAGAN** banneri va fayl nomida `_DRAFT` qo‘shimchasi bo‘ladi.

**AI dan so‘rash** yorlig‘i joriy topilmalar va hisobot kontekstida mahalliy til modeliga savol berish imkonini beradi. Uning javoblari faqat ma’lumot uchun, tashxis emas; siz yozgan hech narsa kompyuterdan tashqariga chiqmaydi.

---

## 8. Hisobotni imzolash

Imzolash ilova tomonidan emas, **AI serveri tomonidan** bajariladi, shuning uchun kompyuter keyinchalik qayta o‘rnatilsa ham u qayd etilgan bo‘ladi.

1. Imzolamoqchi bo‘lgan tilni oching, matnni tekshiring va **✓ Hisobotni imzolash** tugmasini bosing.
2. Server bitta tranzaksiyada saqlaydi: **imzolagan shaxs** (sizning hisob qaydnomangiz), **sana va vaqt**, aniq hisobot matnining **SHA-256 xeshi**, u asoslangan AI qoralamasi, AI topilmalari va ularni yaratgan **model barmoq iz(lar)i**. Audit jurnaliga yozuv kiritiladi.
3. Matn faqat o‘qish uchun bo‘lib qoladi (*Imzolangan va qulflangan*). Panelda *Imzolagan <ism> · <sana/vaqt>* va to‘liq SHA-256 ko‘rsatiladi.
   [Skrinshot: imzolagan shaxs, vaqt tamg‘asi va SHA-256 bilan imzolangan hisobot bloki]

Qoidalar:

- **Har bir tekshiruv va til uchun bitta imzolangan hisobot.** Ikkinchi urinish *Bu tildagi hisobot allaqachon imzolangan* xabarini qaytaradi. Imzolangan hisobotni tahrirlab yoki qayta imzolab bo‘lmaydi. Tuzatish kerak bo‘lsa, ilovadan tashqarida bo‘limingizning qo‘shimcha (addendum) tartibiga amal qiling va IT xizmatiga xabar bering.
- Xuddi shu tekshiruvni ikkinchi tilda ham imzolash mumkin; bu alohida imzolangan yozuv bo‘ladi.
- Faqat **rentgenolog shifokor** va **administrator** rollari imzolay oladi.
- Tizimga o‘z nomingizdan kirgan bo‘lishingiz kerak: imzo — bu sizning hisob qaydnomangiz.

---

## 9. PDF eksport qilish

1. «Hisobot» yorlig‘ida **Imzolangan PDF eksporti** (yoki imzolanmagan qoralama uchun **Qoralama PDF eksporti**) tugmasini bosing. Bir nechta ochilgan tilni bitta hujjatga eksport qilish uchun eksport rejimini *Ochilgan tillar* ga o‘tkazing.
2. Saqlash joyini tanlang (standart — «Yuklab olishlar» papkangiz). So‘ngra fayl Explorer / Finder da ko‘rsatiladi.

PDF tarkibi: klinika sarlavhasi (Sozlamalardagi nom, manzil, telefon), bemor ID va tekshiruv sanasi, hisobot matni, holat belgilari bilan AI topilmalari, SHA-256 barmoq izlari va holati bilan model identifikatorlari, qaror chegarasi, ilova versiyasi, ogohlantirish, hamda yoki **imzo bloki** (imzolagan shaxs, sana/vaqt, to‘liq SHA-256), yoki qizil **AI QORALAMASI — IMZOLANMAGAN** qatori. Fayl nomi: `<bemor yoki tekshiruv id>_<tekshiruv sanasi>_<til>[_DRAFT].pdf`.

Kirill va o‘zbek matni to‘g‘ridan-to‘g‘ri chop etiladi.

---

## 10. AI serveri mavjud bo‘lmaganda

Ilova AI serverini har 15 soniyada so‘rab turadi. Yuqoridagi qizil banner holatni bildiradi:

| Banner | Ma’nosi | Nima qilish kerak |
|---|---|---|
| **AI serveri ishlamayapti — IT xizmatiga murojaat qiling** | Ushbu kompyuterdagi server to‘xtagan yoki unga ulanib bo‘lmayapti. | Yuklash va imzolash o‘chirilgan. Ish stoli qobig‘i serverni avtomatik ravishda 3 martagacha qayta ishga tushiradi; u baribir ishlamasa, dialog oynasi jurnal faylini ko‘rsatadi. Tekshiruvlarni odatdagi ko‘ruvchingizda o‘qing va IT xizmatiga murojaat qiling (qo‘llab-quvvatlash kontakti bannerda va kirish ekranida ko‘rsatiladi). Saqlangan hech narsa yo‘qolmaydi. |
| **AI serveri qisman ishlamoqda: tasdiqlangan model yuklanmadi — IT xizmatiga murojaat qiling** | Server ishlayapti, lekin **tasdiqlangan triaj modeli** yo‘q. Hozir olingan har qanday natija faqat *tasdiqlash kutilayotgan* detektorlardan keladi. | IT xizmati tuzatmaguncha AI natijasidan ustuvorlik berish uchun foydalanmang. |
| **AI serveri xato haqida xabar bermoqda — IT xizmatiga murojaat qiling** | Ichki xato. | IT xizmatiga murojaat qiling. |
| *Shablon hisobot — AI yordamchisi mavjud emas* (bildirishnoma) | Mahalliy til modeli (Ollama) o‘chirilgan. Topilmalarga ta’sir qilmaydi; hisobot qoralamasi shablon bo‘ladi va *AI dan so‘rash* mavjud emas. | Shablonni kerakli tarzda tahrirlang; IT xizmatiga xabar bering. |

Agar yuklash *Tahlil bajarilmadi — IT xizmatiga murojaat qiling* xabari bilan tugasa, tekshiruv saqlanmaydi; keyinroq qayta urinib ko‘ring.

---

## 11. Kimga murojaat qilish kerak

- **Klinika IT xizmati / administrator** — kirish, litsenziya, server va PDF muammolari bo‘yicha birinchi manzil. Ularning kontakti kirish ekranida va har bir xato bannerida ko‘rsatiladi (Sozlamalar → Klinika → Qo‘llab-quvvatlash kontakti orqali sozlanadi).
- **Yetkazib beruvchi** — SIAA Medical AI, Toshkent · www.siaa.uz — xizmat ko‘rsatish shartnomasida ko‘rsatilgan kontakt orqali: model yangilanishlari, litsenziyalar va AI xatosi gumoni bo‘yicha.
- **Klinik masalalar** (o‘tkazib yuborilgan yoki yolg‘on AI belgisi) — tekshiruv ID sini («Tafsilotlar» yorlig‘i → Tekshiruv), o‘z bahoingizni va model barmoq izini qayd eting va pilot rahbari / bo‘lim mudiriga xabar bering. Aynan shu fikr-mulohaza modellarni tasdiqlash va yaxshilash imkonini beradi.
