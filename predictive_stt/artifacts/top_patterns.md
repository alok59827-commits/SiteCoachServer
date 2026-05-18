# Top STT Confusion Patterns

Source: **12198** pair rows from `corpus.csv` + `STT_Error_Dictionary.txt`.

## Top 20 phrase-level mishearings (Deepgram heard → should be)

| # | Heard | Should be | Count |
|---|---|---|---|
| 1 | `शरीर` | `सरिया` | 425 |
| 2 | `डस्टिंग` | `डक्टिंग` | 196 |
| 3 | `गड़बड़` | `गर्डर` | 193 |
| 4 | `कीवी` | `केवी` | 177 |
| 5 | `कंप्यूटर` | `कंप्रेसर` | 160 |
| 6 | `पी डब्लू डी` | `पीडब्ल्यूडी` | 157 |
| 7 | `कम एक्शन` | `कम्पेक्शन` | 145 |
| 8 | `रूलर` | `रोलर` | 141 |
| 9 | `स्टटरिंग` | `शटरिंग` | 139 |
| 10 | `इस फाल्ट` | `एस्फाल्ट` | 132 |
| 11 | `मैं` | `महीने` | 130 |
| 12 | `रूम` | `बूम` | 124 |
| 13 | `पोकलेन` | `पॉलिसी` | 124 |
| 14 | `डांस फार्मर` | `ट्रांसफार्मर` | 122 |
| 15 | `टायर्स` | `टाइल्स` | 121 |
| 16 | `पावर` | `पेवर` | 120 |
| 17 | `सब्जी स्टेशन` | `सबस्टेशन` | 118 |
| 18 | `चिल्लर` | `चिलर` | 118 |
| 19 | `पापा` | `पाइप` | 117 |
| 20 | `डबलू बीम` | `डब्लू बी एम` | 115 |

## Top 20 Devanagari syllable confusions (actual → heard)

| # | Actual | Most-likely heard | P |
|---|---|---|---|
| 1 | `मं` | `` | 100.00% |
| 2 | `छु` | `शु` | 100.00% |
| 3 | `नु` | `ह` | 100.00% |
| 4 | `़ू` | `` | 100.00% |
| 5 | `मुं` | `बू` | 100.00% |
| 6 | `सू` | `सो` | 100.00% |
| 7 | `शं` | `` | 100.00% |
| 8 | `D` | `` | 100.00% |
| 9 | `L` | `` | 100.00% |
| 10 | `छे` | `च्छे` | 100.00% |
| 11 | `निं` | `` | 100.00% |
| 12 | `वो` | `बू` | 100.00% |
| 13 | `गै` | `` | 100.00% |
| 14 | `T` | `` | 100.00% |
| 15 | `u` | `` | 100.00% |
| 16 | `b` | `` | 100.00% |
| 17 | `w` | `` | 100.00% |
| 18 | `l` | `` | 100.00% |
| 19 | `डा` | `o` | 100.00% |
| 20 | `र्शि` | `र` | 100.00% |

## Top 20 character-level confusions (actual → heard)

| # | Actual | Most-likely heard | P |
|---|---|---|---|
| 1 | `ॅ` | `a` | 100.00% |
| 2 | `Z` | `j` | 100.00% |
| 3 | `b` | `` | 100.00% |
| 4 | `x` | `c` | 100.00% |
| 5 | `२` | `म` | 100.00% |
| 6 | `J` | `j` | 96.00% |
| 7 | `g` | `` | 84.62% |
| 8 | `P` | `p` | 81.87% |
| 9 | `U` | `u` | 77.78% |
| 10 | `V` | `v` | 76.67% |
| 11 | `C` | `c` | 76.60% |
| 12 | `A` | `a` | 76.00% |
| 13 | `l` | `` | 75.00% |
| 14 | `w` | `` | 75.00% |
| 15 | `d` | `` | 75.00% |
| 16 | `f` | `` | 75.00% |
| 17 | `h` | `` | 75.00% |
| 18 | `T` | `t` | 74.71% |
| 19 | `M` | `m` | 72.94% |
| 20 | `r` | `` | 72.15% |

## Top 3 phrase confusions per domain

| Domain | # | Heard | Should be | Count |
|---|---|---|---|---|
| 3.85/location hallucinations | 1 | `हुजूर` | `हुज़ूर` | 1 |
| accounts | 1 | `kharacha pani` | `खर्चा पानी` | 1 |
| accounts | 2 | `cash me` | `कैश में` | 1 |
| accounts | 3 | `company` | `कंपनी` | 1 |
| admin/slang | 1 | `rishwat` | `रिश्वत` | 1 |
| admin/slang | 2 | `suwidha shulk` | `सुविधा शुल्क` | 1 |
| admin/slang | 3 | `kharcha pani` | `खर्चा पानी` | 1 |
| administration | 1 | `a d h r card copy` | `आधार कार्ड कॉपी` | 1 |
| administrative | 1 | `abe complain ka entry register me k` | `शिकायत की प्रविष्टि रजिस्टर में करो` | 1 |
| approval | 1 | `complesan cerfit` | `कम्प्लीशन सर्टिफिकेट` | 1 |
| approval | 2 | `authority inspection` | `अथॉरिटी इंस्पेक्शन` | 1 |
| architecture | 1 | `o t yes ka shaft` | `OTS का शाफ़्ट` | 1 |
| architecture | 2 | `tuh b a ka naksha` | `BHK का नक्शा` | 1 |
| architecture | 3 | `front alivashan` | `फ्रंट एलिवेशन` | 1 |
| automobile | 1 | `kalach plet` | `क्लच प्लेट` | 1 |
| automobile | 2 | `shokar leak` | `शॉकर लीक` | 1 |
| automobile | 3 | `aloe vhel` | `अलॉय व्हील` | 1 |
| banking | 1 | `c ibil sckor` | `सिबिल स्कोर` | 1 |
| banking | 2 | `prsnl lon` | `पर्सनल लोन` | 1 |
| banking | 3 | `aemi kitni` | `EMI कितनी` | 1 |
| billing | 1 | `मैं` | `महीने` | 59 |
| billing | 2 | `एम बी` | `एमबी` | 36 |
| billing | 3 | `टीचर` | `ठेकेदार` | 24 |
| bridge_work | 1 | `अपार्टमेंट` | `अबटमेंट` | 20 |
| bridge_work | 2 | `गड़बड़` | `गर्डर` | 20 |
| bridge_work | 3 | `वहां` | `वहाँ` | 2 |
| building | 1 | `शरीर` | `सरिया` | 84 |
| building | 2 | `बीवियां दिए` | `बाइंडिंग वायर` | 50 |
| building | 3 | `टायर्स` | `टाइल्स` | 35 |
| building_construction | 1 | `आलसी सी` | `आरसीसी` | 26 |
| building_construction | 2 | `टायर्स` | `टाइल्स` | 24 |
| building_construction | 3 | `प्लास्टिक` | `प्लास्टर` | 19 |
| carpentry | 1 | `flas dor` | `फ्लश डोर` | 2 |
| carpentry | 2 | `fawikal dabba` | `फेविकोल डब्बा` | 2 |
| carpentry | 3 | `kabjey dhile hai` | `कब्ज़े ढीले हैं` | 1 |
| catering/event | 1 | `tant haus` | `टेंट हाउस` | 1 |
| catering/event | 2 | `pandaal` | `पंडाल` | 1 |
| catering/event | 3 | `hlwai blao` | `हलवाई बुलाओ` | 1 |
| civil | 1 | `स्पीकर` | `Speaker` | 17 |
| civil | 2 | `लॉक` | `ब्लॉक` | 2 |
| civil | 3 | `सीमेंट` | `पेमेंट` | 2 |
| civil work | 1 | `Exhibition` | `एक्सावेशन Excavation` | 1 |
| civil work | 2 | `Costing` | `कास्टिंग Casting` | 1 |
| civil work | 3 | `Shattering` | `शटरिंग Shuttering` | 1 |
| commercial | 1 | `bording pass` | `होर्डिंग पास` | 1 |
| complaint | 1 | `कौन` | `बोल रहा हूँ` | 1 |
| complaint | 2 | `कॉफी` | `हो भी` | 1 |
| complaint | 3 | `आदमी` | `आदेश आगे` | 1 |
| complaint_call | 1 | `Hindrans` | `hindrance` | 1 |
| complaint_call | 2 | `हिंद्रांस` | `Hindrance` | 1 |
| complaint_call | 3 | `खामाखडा` | `खामाखड़ा` | 1 |
| construction | 1 | `kavar bhlok` | `कवर ब्लॉक` | 2 |
| construction | 2 | `colam farma` | `कॉलम फर्मा` | 2 |
| construction | 3 | `dpc dalna` | `DPC डालना` | 2 |
| consumer | 1 | `I have called you regarding launchi` | `मैंने आपको फ्लिपकार्ट पर आपका व्यवस` | 1 |
| consumer | 2 | `It is not logging in` | `यह लॉगिन नहीं हो रहा है` | 1 |
| consumer | 3 | `your pickup PIN code has not been v` | `आपका पिकअप पिन कोड वेरिफाई नहीं हुआ` | 1 |
| contract | 1 | `sq fut me` | `स्क्वायर फुट में` | 1 |
| contract | 2 | `lamsam dena hai` | `लम्पसम देना है` | 1 |
| contract | 3 | `pnalty lagegi` | `पेनल्टी लगेगी` | 1 |
| defects | 1 | `kam bigad` | `काम बिगाड़` | 1 |
| defects | 2 | `tod fod` | `तोड़ फोड़` | 1 |
| defects | 3 | `r wrak` | `रीवर्क` | 1 |
| dg set | 1 | `abe dg set kal sirf do ghantak chan` | `डीजी सेट कल केवल दो घंटे चले फिर बं` | 1 |
| digital | 1 | `लोग इन` | `लॉगिन` | 51 |
| digital | 2 | `सर अंदर` | `सरेंडर` | 46 |
| digital | 3 | `हुजूर` | `हुज़ूर` | 8 |
| documentation | 1 | `engineering` | `इंजीनियरिंग` | 1 |
| documentation | 2 | `haan perspective ka photo upload ma` | `हाँ परपेक्ष वाला फोटो अपलोड मत करो ` | 1 |
| documentation | 3 | `sir jeman me labor ka attendance au` | `सर जेएमआर में मजदूरों की उपस्थिति औ` | 1 |
| earthwork | 1 | `जे एस बी` | `जेसीबी` | 21 |
| earthwork | 2 | `जय सी बी` | `जेसीबी` | 20 |
| earthwork | 3 | `हाई बार` | `हाइवा` | 20 |
| education | 1 | `आई टी यू एस` | `आईटीयूएस` | 1 |
| education | 2 | `टालेंट` | `टैलेंट` | 1 |
| education | 3 | `अल्टिमेट` | `अल्टीमेट` | 1 |
| electrical | 1 | `konsil light` | `कंसील लाइट` | 2 |
| electrical | 2 | `main suvich` | `मेन स्विच` | 2 |
| electrical | 3 | `kabel tre` | `केबल ट्रे` | 2 |
| electrical/it | 1 | `wifi ka rutr` | `WiFi का राउटर` | 1 |
| electricity | 1 | `कीवी` | `केवी` | 95 |
| electricity | 2 | `डांस फार्मर` | `ट्रांसफार्मर` | 67 |
| electricity | 3 | `सब्जी स्टेशन` | `सबस्टेशन` | 62 |
| electronics | 1 | `kombo dalga` | `कॉम्बो डलेगा` | 1 |
| electronics | 2 | `fldar ukhad` | `फोल्डर उखड़` | 1 |
| electronics | 3 | `chrjng pin` | `चार्जिंग पिन` | 1 |
| elevator | 1 | `lifft safat` | `लिफ्ट शाफ़्ट` | 1 |
| elevator | 2 | `apit me paani` | `पिट में पानी` | 1 |
| elevator | 3 | `msin rum` | `मशीन रूम` | 1 |
| estimation | 1 | `haan chaudu surupi wala estimate JE` | `हाँ वाला अनुमान आज ही जेई साहब को भ` | 1 |
| excavation | 1 | `extra vision` | `एक्सावेशन` | 2 |
| excavation | 2 | `jcb bula lo` | `JCB बुला लो` | 1 |
| excavation | 3 | `faundashan ka gadda` | `फाउंडेशन का गड्ढा` | 1 |
| fabrication | 1 | `khidki ki greel` | `खिड़की की ग्रिल` | 1 |
| fabrication | 2 | `relig lagani hai` | `रेलिंग लगानी है` | 1 |
| fabrication | 3 | `shatar lagana` | `शटर लगाना` | 1 |
| family | 1 | `है` | `हैं` | 4 |
| family | 2 | `पती` | `पति` | 3 |
| family | 3 | `जाएं` | `जाएँ` | 2 |
| farming/mandi | 1 | `kheti badi` | `खेती बाड़ी` | 1 |
| farming/mandi | 2 | `kisan bhai` | `किसान भाई` | 1 |
| farming/mandi | 3 | `trctar trli` | `ट्रैक्टर ट्रॉली` | 1 |
| finance | 1 | `abe ye fad giri mat karo actual met` | `यह ठगगिरी मत करो वास्तविक मीटर से अ` | 1 |
| finance | 2 | `abe pan cha hazar ka diesel aur ala` | `लगभग हजार का डीजल और अलग से मजदूरी ` | 1 |
| finance | 3 | `diesel` | `डीजल` | 1 |
| financial | 1 | `you will have to pay for the compan` | `आपको कंपनी के अकाउंट में भुगतान करन` | 1 |
| financial | 2 | `आधार` | `आठ` | 1 |
| financial | 3 | `रुपे` | `रुपये` | 1 |
| finishing | 1 | `khidki ki greel` | `खिड़की` | 1 |
| finishing | 2 | `painting` | `ग्रिल की पेंटिंग` | 1 |
| finishing | 3 | `color fad` | `कलर फेड` | 1 |
| fire safety | 1 | `far darwaja` | `फायर डोर` | 1 |
| fire safety | 2 | `hydent pib` | `हाइड्रेंट पाइप` | 1 |
| fire safety | 3 | `saprinkalar` | `स्प्रिंकलर` | 1 |
| flooring | 1 | `marbal lagana` | `मार्बल लगाना` | 1 |
| flooring | 2 | `grenaite top` | `ग्रेनाइट टॉप` | 1 |
| flooring | 3 | `koota ston` | `कोटा स्टोन` | 1 |
| general | 1 | `है` | `हैं` | 3 |
| general | 2 | `thik jama` | `ठीक है` | 2 |
| general | 3 | `left side me thik` | `लेफ्ट साइड में` | 2 |
| government | 1 | `kagaz bhejo mb` | `कागज भेजो MB` | 1 |
| green building | 1 | `solar panl` | `सोलर पैनल` | 1 |
| green building | 2 | `on garid` | `ऑन ग्रिड` | 1 |
| green building | 3 | `aaf garid` | `ऑफ ग्रिड` | 1 |
| hardware | 1 | `al drop` | `एल्ड्रॉप` | 2 |
| hardware | 2 | `screw` | `स्क्रू` | 1 |
| heavy_machinery | 1 | `रूम` | `बूम` | 66 |
| heavy_machinery | 2 | `पावर ट्रेन` | `टावर क्रेन` | 48 |
| heavy_machinery | 3 | `मैचिंग` | `बैचिंग` | 44 |
| hvac | 1 | `चिल्लर` | `चिलर` | 66 |
| hvac | 2 | `आज यू` | `एएचयू` | 55 |
| hvac | 3 | `कंप्यूटर` | `कंप्रेसर` | 54 |
| hydro | 1 | `दाई` | `भाई` | 1 |
| hydro | 2 | `तालमाओ का` | `टालमटोल कर` | 1 |
| hydro | 3 | `चाखथे` | `चाहते हुए भी` | 1 |
| hydrotest | 1 | `abe gap losing jaldi karo fir hydro` | `गैप क्लोजिंग जल्दी पूरा करो फिर हाइ` | 1 |
| hydrotest | 2 | `अफ एल` | `एफई` | 1 |
| hydrotest | 3 | `बोला` | `कहा है कि` | 1 |
| industrial | 1 | `p e b shad` | `PEB शेड` | 1 |
| industrial | 2 | `pari enjinierad` | `प्री इंजीनियर्ड` | 1 |
| industrial | 3 | `ai bheem` | `आई बीम` | 1 |
| inspection | 1 | `sir sath giri karke purana valve na` | `सर ठगगिरी करके पुराना वाल्व नया बता` | 1 |
| inspection | 2 | `sir f l engineer bolra valve chambe` | `सर एफई इंजीनियर कह रहे हैं कि वाल्व` | 1 |
| inspection | 3 | `f l` | `एफई` | 1 |
| insurance | 1 | `पोकलेन` | `पॉलिसी` | 60 |
| insurance | 2 | `ही शायद` | `शुक्रवार` | 35 |
| insurance | 3 | `हुजूर` | `हुज़ूर` | 7 |
| interior | 1 | `jipsam board` | `जिप्सम बोर्ड` | 1 |
| interior | 2 | `fal siling` | `फॉल सीलिंग` | 1 |
| interior | 3 | `pop ki siling` | `POP की सीलिंग` | 1 |
| irrigation | 1 | `गैराज` | `बैराज` | 43 |
| irrigation | 2 | `टाइम` | `डैम` | 43 |
| irrigation | 3 | `कमाल` | `कनाल` | 40 |
| it | 1 | `bag aa gya` | `बग आ गया` | 1 |
| it | 2 | `di bag kro` | `डीबग करो` | 1 |
| it | 3 | `sorce cod` | `सोर्स कोड` | 1 |
| jal nigam | 1 | `c pvc ppe` | `CPVC पाइप` | 1 |
| jal nigam | 2 | `overhad tank` | `ओवरहेड टैंक` | 1 |
| jal nigam | 3 | `insepcsn chambr` | `इंस्पेक्शन चेंबर` | 1 |
| kitchen | 1 | `chimani ka houl` | `चिमनी का होल` | 1 |
| kitchen | 2 | `moduler kichen` | `मॉड्यूलर किचन` | 1 |
| labor | 1 | `aaj lebara nahi aayi` | `आज लेबर नहीं आई` | 4 |
| labor | 2 | `मैं` | `महीने` | 2 |
| labor | 3 | `thekedar ko paise diye` | `ठेकेदार को पैसे दिए` | 1 |
| labor_management | 1 | `शॉपर वाइजर` | `सुपरवाइजर` | 2 |
| labor_management | 2 | `पी मिंट` | `पेमेंट` | 2 |
| labor_management | 3 | `arey Shopper visor hajri lagao Pea ` | `अरे सुपरवाइजर हाज़िरी लगाओ पेमेंट आ` | 1 |
| labour | 1 | `dihari` | `दिहाड़ी` | 1 |
| labour | 2 | `hajri rjistr` | `हाज़िरी रजिस्टर` | 1 |
| labour | 3 | `beldar` | `बेलदार` | 1 |
| land records | 1 | `khsra khtoni` | `खसरा खतौनी` | 1 |
| land records | 2 | `copy office` | `कॉपी ऑफिस` | 1 |
| land records | 3 | `sir file` | `सर फाइल` | 1 |
| layout | 1 | `chuna markng` | `चूना मार्किंग` | 1 |
| layout | 2 | `guniya mila lo` | `गुनिया मिला लो` | 1 |
| legal | 1 | `हुजूर` | `हुज़ूर` | 8 |
| legal | 2 | `जगह रहा छब्बीस` | `अजगरहा` | 8 |
| legal | 3 | `Friday ही शायद` | `शुक्रवार` | 5 |
| legal/police | 1 | `fir huj` | `FIR दर्ज` | 1 |
| legal/police | 2 | `cmpalint likhi` | `कंप्लेंट लिखी` | 1 |
| legal/police | 3 | `steson incharj` | `स्टेशन इंचार्ज` | 1 |
| location | 1 | `हुजूर` | `हुज़ूर` | 1 |
| location | 2 | `Friday ही शायद` | `शुक्रवार` | 1 |
| location | 3 | `जगह रहा छब्बीस` | `अजगरहा` | 1 |
| logistics | 1 | `tactor trolley` | `ट्रैक्टर ट्रॉली` | 1 |
| logistics | 2 | `bhaari vahan` | `भारी वाहन` | 1 |
| logistics | 3 | `no antar` | `नो एंट्री` | 1 |
| logistics/railway | 1 | `trasport me` | `ट्रांसपोर्ट में` | 1 |
| logistics/railway | 2 | `bilti bhej` | `बिल्टी भेज` | 1 |
| logistics/railway | 3 | `ael aar copy` | `LR कॉपी` | 1 |
| machinery | 1 | `जूसी` | `जेसीबी` | 23 |
| machinery | 2 | `पोक लेन` | `पोकलेन` | 23 |
| machinery | 3 | `हाईवे` | `हाइवा` | 23 |
| maintenance | 1 | `kalai karni` | `कलई करनी` | 1 |
| maintenance | 2 | `overhad tank` | `ओवरहेड टैंक` | 1 |
| management | 1 | `rojana karcha` | `रोज़ाना खर्चा` | 1 |
| management | 2 | `advance pement` | `एडवांस पेमेंट` | 1 |
| management | 3 | `bill cleyar` | `बिल क्लियर` | 1 |
| manufacturing | 1 | `ml aagya` | `माल आ गया` | 1 |
| manufacturing | 2 | `ro metral` | `रॉ मटेरियल` | 1 |
| manufacturing | 3 | `asmbly lin` | `असेंबली लाइन` | 1 |
| masonry | 1 | `bal kitne inch ki` | `वॉल Wall कितने इंच की` | 1 |
| masonry | 2 | `palastar kab hoga` | `प्लास्टर कब होगा` | 1 |
| masonry | 3 | `tyles lagani hai` | `टाइल्स लगानी है` | 1 |
| material | 1 | `a c c sment` | `ACC सीमेंट` | 3 |
| material | 2 | `krashar dust` | `क्रेशर डस्ट` | 2 |
| material | 3 | `bees gori bhejwa dijiye` | `बीस बोरी भिजवा दीजिये` | 1 |
| measurement | 1 | `Square meter` | `स्क्वायर मीटर Sqm` | 1 |
| measurement | 2 | `Cupic` | `क्यूबिक मीटर Cum` | 1 |
| measurement | 3 | `Feed` | `फीट Feet` | 1 |
| mechanical | 1 | `room` | `रूम` | 1 |
| mechanical | 2 | `choked` | `choke` | 1 |
| medical | 1 | `oop d` | `OPD` | 1 |
| medical | 2 | `i p di` | `IPD` | 1 |
| medical | 3 | `ay se u` | `ICU` | 1 |
| metro | 1 | `गड़बड़` | `गर्डर` | 13 |
| metro | 2 | `ऑनलाइन मेंट` | `अलाइनमेंट` | 8 |
| metro | 3 | `इसका फोल्डिंग` | `स्कैफोल्डिंग` | 4 |
| mining | 1 | `प्लास्टिक` | `ब्लास्टिंग` | 50 |
| mining | 2 | `डीलिंग` | `ड्रिलिंग` | 44 |
| mining | 3 | `साफ़` | `शाफ़्ट` | 43 |
| mixed | 1 | `टीचर` | `ठेकेदार` | 18 |
| mixed | 2 | `एक्स्ट्रा विज़न` | `एक्सावेशन` | 17 |
| mixed | 3 | `एम बी` | `एमबी` | 15 |
| nan | 1 | `The bill that you will select will ` | `आपने जो चुना है` | 1 |
| nan | 2 | `we request you to wait for some tim` | `कृपया कुछ समय प्रतीक्षा करें` | 1 |
| nan | 3 | `देपक अपने सत्यम लाइन में है` | `दीपक` | 1 |
| ohsr_work | 1 | `समरसेब` | `सबमर्सिबल` | 2 |
| ohsr_work | 2 | `पीच ही` | `पीएचई` | 2 |
| ohsr_work | 3 | `वाटर प्रूफ` | `वाटरप्रूफ` | 2 |
| painting | 1 | `pu t karwa do` | `पुट्टी करवा दो` | 1 |
| painting | 2 | `praymar bacha hai` | `प्राइमर बचा है` | 1 |
| painting | 3 | `tar pis ka tel` | `तारपीन का तेल` | 1 |
| people | 1 | `Labor` | `लेबर Labour` | 1 |
| permission | 1 | `haan govind nagar wale route pe roa` | `हाँ गोविंदगढ़ वाले मार्ग पर सड़क कट` | 1 |
| pipeline | 1 | `पिन` | `पाइप लाइन` | 29 |
| pipeline | 2 | `बॉल` | `वाल्व` | 22 |
| pipeline | 3 | `जलनाम` | `जल निगम` | 21 |
| plumbing | 1 | `sr pipe` | `SWR पाइप` | 2 |
| plumbing | 2 | `kansil pipe` | `कंसील पाइप` | 2 |
| plumbing | 3 | `sink ka sayfon` | `सिंक का साइफन` | 1 |
| printing/studio | 1 | `falex print` | `फ्लेक्स प्रिंट` | 1 |
| printing/studio | 2 | `banr lga` | `बैनर लगा` | 1 |
| printing/studio | 3 | `vistng card` | `विज़िटिंग कार्ड` | 1 |
| process | 1 | `सर अंदर` | `सरेंडर` | 4 |
| process | 2 | `जगह रहा छब्बीस` | `अजगरहा` | 3 |
| process | 3 | `Friday ही शायद` | `शुक्रवार` | 2 |
| project | 1 | `Pump house` | `पंप हाउस` | 1 |
| project | 2 | `Distribution` | `डिस्ट्रीब्यूशन` | 1 |
| project status | 1 | `sir post close ka file abhi tak off` | `सर फोरक्लोज़र की फाइल अभी तक कार्या` | 1 |
| property | 1 | `मंगला` | `बंगला` | 1 |
| public grievance | 1 | `ca meltline me` | `CM हेल्पलाइन में` | 1 |
| public issue | 1 | `compalint` | `कंप्लेंट` | 1 |
| quality | 1 | `lab` | `लैब` | 2 |
| quality | 2 | `compksn test` | `कम्पैक्शन टेस्ट` | 1 |
| quality | 3 | `compksn test report` | `कम्पैक्शन टेस्ट रिपोर्ट` | 1 |
| railways | 1 | `गड़बड़` | `गर्डर` | 53 |
| railways | 2 | `ब्लास्ट` | `बैलास्ट` | 51 |
| railways | 3 | `ऑनलाइन मेंट` | `अलाइनमेंट` | 38 |
| rcc_work | 1 | `स्टटरिंग` | `शटरिंग` | 57 |
| rcc_work | 2 | `वाई ब्रेटर` | `वाइब्रेटर` | 34 |
| rcc_work | 3 | `कॉन क्रिएट` | `कंक्रीट` | 33 |
| re | 1 | `tu bech ke` | `BHK` | 1 |
| re | 2 | `brokeraj kitni` | `ब्रोकरेज कितनी` | 1 |
| re | 3 | `bayaana de diya` | `बयाना दे दिया` | 1 |
| reinforcement | 1 | `sariya` | `सरिया` | 1 |
| reinforcement | 2 | `binding` | `बाइंडिंग` | 1 |
| rental | 1 | `bijli bill alag` | `बिजली बिल अलग` | 2 |
| rental | 2 | `kidaaye par dena hai` | `किराये पर देना है` | 1 |
| rental | 3 | `dipojit amount` | `डिपॉज़िट अमाउंट` | 1 |
| repair | 1 | `c c roof` | `RCC रूफ` | 2 |
| repair | 2 | `epoksi karni` | `इपॉक्सी करनी` | 2 |
| repair | 3 | `sir` | `सर` | 2 |
| resource management | 1 | `haan risorses kam hai isliye road r` | `हाँ रिसोर्सेज कम हैं इसलिए सड़क पुन` | 1 |
| road_work | 1 | `कम एक्शन` | `कम्पेक्शन` | 74 |
| road_work | 2 | `रूलर` | `रोलर` | 72 |
| road_work | 3 | `इस फाल्ट` | `एस्फाल्ट` | 67 |
| roadwork | 1 | `site` | `साइट` | 2 |
| roadwork | 2 | `compaktor chalao` | `कम्पैक्टर चलाओ` | 2 |
| roadwork | 3 | `kafi kaam baaki hai road restorसन k` | `काफी काम बाकी है रोड रेस्टोरेशन का ` | 1 |
| roofing | 1 | `p v c sheet` | `PVC शीट` | 1 |
| roofing | 2 | `fivar sheet` | `फाइबर शीट` | 1 |
| roofing | 3 | `san shed` | `सनशेड` | 1 |
| safety | 1 | `gam boot` | `गम बूट` | 2 |
| safety | 2 | `kambal` | `कम्बल` | 2 |
| safety | 3 | `safty baelt` | `सेफ्टी बेल्ट` | 1 |
| sanitary | 1 | `vas basin` | `वॉश बेसिन` | 1 |
| sanitary | 2 | `pedstal basin` | `पेडस्टल बेसिन` | 1 |
| sanitary | 3 | `comod sheet` | `कमोड सीट` | 1 |
| security | 1 | `c c camera` | `CCTV कैमरा` | 2 |
| security | 2 | `wifi routar` | `WiFi राउटर` | 2 |
| security | 3 | `tar bandi` | `तार बंदी` | 1 |
| sentence | 1 | `शरीर` | `सरिया` | 213 |
| sentence | 2 | `गड़बड़` | `गर्डर` | 93 |
| sentence | 3 | `डस्टिंग` | `डक्टिंग` | 83 |
| sewerage | 1 | `Friday ही शायद` | `शुक्रवार को` | 1 |
| sewerage | 2 | `Friday ही शायद` | `शुक्रवार` | 1 |
| sewerage | 3 | `जगह रहा छब्बीस` | `अजगरहा` | 1 |
| shuttering | 1 | `sir pan cha hazar me shuttering ka ` | `सर हजार में शटरिंग का सामान उपलब्ध ` | 1 |
| site issues | 1 | `paani bhar gya` | `पानी भर गया` | 1 |
| site issues | 2 | `light kat gai` | `लाइट कट गई` | 1 |
| site issues | 3 | `lebar bhag gai` | `लेबर भाग गई` | 1 |
| site_call | 1 | `office हूं sir` | `ऑफिस में हूँ सर` | 1 |
| site_call | 2 | `zedambier` | `ज़ेडम्बियर की` | 1 |
| site_call | 3 | `लडंबियार` | `ज़ेडम्बियर` | 1 |
| smart home | 1 | `smart svich` | `स्मार्ट स्विच` | 1 |
| smart home | 2 | `alaxsa` | `एलेक्सा` | 1 |
| smart home | 3 | `hom automatn` | `होम ऑटोमेशन` | 1 |
| store | 1 | `haan thak giri pakdi gayi store me ` | `हाँ ठगगिरी पकड़ी गई स्टोर में नकली ` | 1 |
| structural | 1 | `expanan joint` | `एक्सपेंशन जॉइंट` | 1 |
| structural | 2 | `block` | `ब्लॉक` | 1 |
| technical | 1 | `haan hindrans clear nahi hua isliye` | `हाँ हिंद्रांस साफ नहीं हुआ इसलिए पा` | 1 |
| telecom | 1 | `डस्टिंग` | `डक्टिंग` | 65 |
| telecom | 2 | `सप्लाई` | `स्प्लाइसिंग` | 48 |
| telecom | 3 | `मेन हॉल` | `मैनहोल` | 45 |
| tools | 1 | `grendar machine` | `ग्राइंडर मशीन` | 1 |
| tools | 2 | `dril bit lao` | `ड्रिल बिट लाओ` | 1 |
| tools | 3 | `hathori lana` | `हथौड़ी लाना` | 1 |
| transport | 1 | `ब्लूरस` | `बोलेरो` | 1 |
| transport | 2 | `वेट` | `इंतजार` | 1 |
| transport | 3 | `ब्लू रॉस` | `बोलेरो` | 1 |
| upvc/glass | 1 | `u p v c vindo` | `UPVC विंडो` | 1 |
| upvc/glass | 2 | `almonyam section` | `एल्युमीनियम सेक्शन` | 1 |
| upvc/glass | 3 | `sliding get` | `स्लाइडिंग गेट` | 1 |
| urdu/arabic | 1 | `पार` | `पार्शियली क्लोज्ड` | 1 |
| valve chamber | 1 | `haan` | `हाँ` | 1 |
| valve chamber | 2 | `लोसिंग ke time purana valve chamber` | `क्लोजिंग के समय पुराने वाल्व चैंबर ` | 1 |
| variation | 1 | `pin` | `पाइप लाइन` | 1 |
| variation | 2 | `Piper line` | `पाइप लाइन` | 1 |
| variation | 3 | `Pine line` | `पाइप लाइन` | 1 |
| water_supply | 1 | `समरसेब` | `सबमर्सिबल` | 51 |
| water_supply | 2 | `पीच ही` | `पीएचई` | 51 |
| water_supply | 3 | `पी ए च` | `पीएचई` | 43 |
| waterproofing | 1 | `c c roof` | `RCC रूफ` | 1 |
| waterproofing | 2 | `tar falt` | `टार फेल्ट` | 1 |
| waterproofing | 3 | `b t main` | `बिटुमेन` | 1 |
| xl | 1 | `The size will be` | `साइज़` | 1 |
| अरे ग्रेडर को बोल सबग्रेड पे मिट्टी डाले और रोलर चला दे। | 1 | `कंप्यूटर` | `कंप्रेसर` | 1 |
| अरे ग्रेडर को बोल सबग्रेड पे मिट्टी डाले और रोलर चला दे। | 2 | `चिल्लर` | `चिलर` | 1 |
| अरे ग्रेडर को बोल सबग्रेड पे मिट्टी डाले और रोलर चला दे। | 3 | `डस्टिंग` | `डक्टिंग` | 1 |
| आपको वीडियो कॉल पर सेलेक्शन दिखाया जाएगा। | 1 | `Sir` | `you will be selected on video call` | 1 |
| आपको सिंगल कुर्ती | 1 | `you will get single kurti` | `piece` | 1 |
| क्या मैं जान सकता हूँ कि आप फ्लिपकार्ट पर कौन से प्रोडक्ट बेचने वाले हैं? | 1 | `Can I know what products you are go` | `sir` | 1 |
| तो अगर नहीं है तो... | 1 | `ने बना रहा शाइड` | `साइड` | 1 |
| पिन कोड 486003 रहेगा। | 1 | `What PIN code will remain` | `sir` | 1 |
| मैं अभी थोड़ा व्यस्त हूँ। क्या मैं आपको एक घंटे बाद कॉल कर सकता हूँ? | 1 | `Sir` | `I am a little out of place right no` | 1 |
| मैं थोड़ा चाह रहा हूँ। | 1 | `परदेशन` | `प्रदेश` | 1 |
| मैं फ्लिपकार्ट से हितेश बोल रहा हूँ। | 1 | `Sir` | `I am Hitesh speaking from Flipkart` | 1 |
| वाइज टाइपिंग और उनकी लोकेशन वगैरह है। | 1 | `डाटा बना` | `डेटा` | 1 |
| वाइज टाइपिंग और उनकी लोकेशन वगैरह है। | 2 | `आपकी पास कितने कहां कहां लोकेशन वाइ` | `है` | 1 |
| हम मीटिंग में नहीं जुड़ पाए हैं। | 1 | `अल्लो हाँ` | `मेटिंग में नहीं जुड़े रहे हैं` | 1 |
