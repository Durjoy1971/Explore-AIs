# `Motive`

`The research focuses on the development of an efficient Bangla SMS spam detection system utilizing BanglaBERT and MobileBERT, which were trained on a curated dataset of authentic Bangla SMS messages comprising both spam and legitimate (ham) classifications. Initially, the BanglaBERT model achieved a validation accuracy of 96%; however, owing to the computational constraints inherent to mobile devices, it was subsequently substituted with MobileBERT. MobileBERT presents a reduced model footprint, accelerated inference capabilities, and an equivalent accuracy rate of 96% while maintaining superior resource efficiency, thereby establishing its suitability for mobile deployment.`

`The fine-tuned MobileBERT architecture is hosted on Hugging Face Spaces and integrated via FastAPI to facilitate real-time SMS processing, ensuring a streamlined and seamless user experience. The বাংলা বার্তা সুরক্ষা application automatically categorizes incoming messages into spam or ham in real-time while incorporating structured message management functionalities. Messages are categorized across two distinct interfaces: one dedicated to verified contacts and recognized senders, and another for unidentified contacts or promotional communications. Users retain the ability to reassign contacts across these categories without altering the underlying algorithmic spam/ham classification.`

`Messages containing text in the Bangla language are processed directly by the model, with spam entries designated in red, legitimate messages in green, and non-Bangla messages highlighted in amber. To optimize operational performance, the application caches processed communications locally using Hive, thereby minimizing redundant API requests. Through its high-precision detection methodology and structured interface, বাংলা বার্তা সুরক্ষা delivers a secure, efficient, and robust SMS management solution for Bangla-speaking users.`

# `Problem Statement`

`Despite the widespread adoption of mobile telecommunications, spam SMS messages remain a persistent challenge, causing significant inconvenience and presenting security risks to mobile users. This issue is particularly critical for Bangla-speaking populations, who lack access to specialized tools capable of effectively identifying and filtering spam communications written in the Bangla language. Existing automated spam detection systems are predominantly configured for globally dominant languages such as English, leaving Bangla speakers underserved and vulnerable to unsolicited or malicious content.`

`In the absence of a dedicated Bangla-specific spam filtering mechanism, users are forced to manually screen incoming messages. This manual process is inefficient, diminishes overall productivity, and increases susceptibility to fraudulent activities, phishing schemes, and misleading communications. Additionally, the manual organization of incoming text messages imposes a substantial burden, particularly for individuals receiving high daily message volumes.`

`To mitigate these issues, there is an imperative requirement for a localized, highly accurate spam detection framework tailored specifically to the Bangla language. Such a system must be computationally efficient, resource-optimized, and suitable for mobile environments to facilitate seamless, real-time integration. The implementation of a reliable Bangla spam detection architecture would enhance digital security, optimize message organization, and fulfill a critical technological gap for Bangla-speaking users.`

* `Built using the Flutter framework, the application targets Android operating systems, offering a refined and efficient user interface.`  
* `Core features encompass the following:`  
  * `Introductory Screen: Outlines system functionality and prompts the user for necessary SMS access authorizations.`  
  * `Message Categorization: Incoming SMS communications are systematically organized into two distinct views:`  
    * `Known Contacts: Displays messages received from verified and saved senders.`  
    * `Promotional/Unknown Contacts: Isolates messages originating from unsaved numbers, promotional entities, and user-restricted senders.`  
  * `Real-Time Classification: Upon selecting an individual message:`  
    * `The message content is transmitted to the Hugging Face Inference API.`  
    * `Evaluations are returned instantaneously, categorizing content into defined designations:`  
      * `Spam (Red): Identifies unwanted, unsolicited, or potentially malicious communications.`  
      * `Ham (Green): Identifies legitimate and safe communications.`  
      * `Non-Bangla (Amber): Flags messages written in alternative scripts or neutral text.`  
* `Performance Optimization:`  
  * `The platform emphasizes high predictive accuracy and minimal inference latency to ensure optimal responsiveness.`

`By facilitating structured message management, the system provides users with enhanced security, control, and efficiency in handling SMS communications.`

# **`REFERENCES`**

`[1] T. Sahmoud and D. M. Mikki, “Spam detection using BERT,” arXiv preprint arXiv:2206.02443, 2022.`

`[2] M. K. Islam, M. Al Amin, M. R. Islam, M. I. Mahbub, M. I. H. Showrov, and C. Kaushal, “Spam detection with comparative analysis and spamming words extractions,” in 2021 9th International Conference on Reliability, Infocom Technologies and Optimization (Trends and Future Directions) (ICRITO), IEEE, 2021, pp. 1–9.`

`[3] R. Amin, M. M. Rahman, and [Author's initials missing], “A Bangla spam email detection and datasets creation approach based on machine learning algorithms,” in 2019 3rd International Conference on Electrical, Computer & Telecommunication Engineering (ICECTE), IEEE, 2019, pp. 169–172.`

`[4] A. Al Maruf, A. Al Human, M. M. Haque, T. T. Jidney, and Z. Aung, “Ensemble approach to classify spam SMS from Bengali text,” in International Conference on Advances in Computing and Data Sciences, Springer, 2023, pp. 440–453.`

`[5] A. K. Mandal and R. Sen, “Supervised learning methods for Bangla web document categorization,” arXiv preprint arXiv:1410.2045, 2014.`

`[6] T. Islam, S. Latif, and N. Ahmed, “Using social networks to detect malicious Bangla text content,” in 2019 1st International Conference on Advances in Science, Engineering and Robotics Technology (ICASERT), Dhaka, Bangladesh, 2019, pp. 1–4, doi: 10.1109/ICASERT.2019.8934841.`

`[7] S. M. M. Hossain, A. Sen, and K. Deb, “Detecting spam SMS using self-attention mechanism,” in Intelligent Computing & Optimization (ICO 2022), Lecture Notes in Networks and Systems, vol. 569, Springer, Cham, 2023. doi: 10.1007/978-3-031-19958-5_17.`

`[8] F. Khan, R. Mustafa, F. Tasnim, T. Mahmud, M. S. Hossain, and K. Andersson, “Exploring BERT and ELMo for Bangla spam SMS dataset creation and detection,” in 2023 26th International Conference on Computer and Information Technology (ICCIT), Cox's Bazar, Bangladesh, 2023, pp. 1–6, doi: 10.1109/ICCIT60459.2023.10441093.`

`[9] T. A. Almeida, J. M. G. Hidalgo, and A. Yamakami, “Contributions to the study of SMS spam filtering: New collection and results,” in Proceedings of the 11th ACM Symposium on Document Engineering (DocEng '11), Association for Computing Machinery, New York, NY, USA, 2011, pp. 259–262. doi: 10.1145/2034691.2034742.`

`[10] A. S. Aski and N. K. Sourati, “Proposed efficient algorithm to filter spam using machine learning techniques,” Pacific Science Review A: Natural Science and Engineering, vol. 18, no. 2, pp. 145–149, 2016, ISSN: 2405-8823. doi: 10.1016/j.psra.2016.09.017.`

