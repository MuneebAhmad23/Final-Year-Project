# Security in Affective Computing

### Abstract

With the boom of machine learning, we can see it enter all faucets of our lives. As it
becomes a more commonly adopted technology in information sharing, so must its
privacy preservation. In this project, we aim to enlighten the reader on some possible
issues that might arise due to the unregulated manner is which machine learning models
handle sensitive data. We discuss prior works done to address this issue, and what could
be done to improve them. Then we attempt to propose a novel method of our own,
discuss our methodology and the results we achieve with our methods. We focus on
image data, and we focus on encrypting said data using a suitable technique. The
datasets we use for this project vary between facial recognition, detection, object
classification as well as written text data. The purpose of using such different datasets
is to gauge the strengths and weaknesses of our proposed methods.
The combined use of encryption with machine learning is very promising, as it allows
for safe integration of ‘artificial intelligence’ with all our day-to-day tasks, as well as
enabling sensitive data like medical or financial records to be used in pattern extraction,
which in turn allows us to have much more sophisticated tools tomorrow.

### CHAPTER 1: INTRODUCTION

#### 1.1 Overview

Machine Learning— what once was a niche field for specialized academics and
scientists, has now become ubiquitous in our everyday lives and a matter of common
knowledge.
This evolution can be credited to the recent boom of artificial intelligence and machine
learning, specifically in emotional processing and comprehension, natural language
processing, etc.,
These advancements have turned machine learning practitioners' attention to
improving model robustness and, perhaps most importantly, optimizing the varying
levels of quality and quantity of data used in the training process.  
The exponential growth in the volume of data required for efficient machine learning
model training has given rise to the name "big data." This transition is the result of
several reasons, the most notable of which are the birth of the digital age, the
development of internet-connected gadgets, and the pervasiveness of online platforms.
As a result, the ‘big data’ landscape is often characterized by the "Three Vs": volume,
variety, and velocity.  
Volume is a fundamental component of big data, referring to the sheer amount of
information created and kept in digital formats today. The greater the dataset available
for analysis, the more effective and precise machine learning models become. This
phenomenon is based on the idea that when data sets expand in size, patterns, and
insights that were previously concealed in smaller samples become visible.  
Variety adds diversity to the data landscape, which increases the potential of big data.
Modern datasets include a variety of unstructured and semi-structured data formats in
addition to standard structured data. These many data sources include sensor data, text,
pictures, videos, and more. These data sources' depth and variety bring up fresh,
frequently surprising ways to infer conclusions and make predictions on how to use and get significant insights from this range of data kinds both a problem and an opportunity in the field of machine learning.
Velocity refers to the rate at which data is created, gathered, and exchanged. The real
time aspect of the modern period magnifies the number, diversity, and velocity of
information created. This is made possible by data streams from mobile phones, IoT
devices, and internet platforms. Instantaneous exchange, analysis, and decision-making
are made possible by this quick data flow. However, it also brings privacy front and
center as a crucial issue of global public policy.
Both enormous potential and significant difficulties are presented by the sheer amount,
diversity, and speed of data created in our digital age. On the one hand, they enable
machine learning algorithms to produce important insights and generate forecasts that
are ever more accurate. On the other hand, they bring up important issues about data
privacy, security, and the ethical handling of personal information.

#### 1.2 Addressing Privacy Concerns

Machine learning has revolutionized many facets of our lives by bringing about a
period of previously unheard-of capabilities and applications. There are, however,
certain limitations to this revolutionary potential, and the degradation of privacy is one
of the most important ones. The astounding developments in facial recognition
technology are a striking example of this issue. It used to be limited to recognizing
hazy pictures of cats, but it quickly developed to attain almost immediate human
recognition. The vast reservoirs of digitized images collected from websites, social
media platforms, license bureaus, security cameras, and many other sources are largely
responsible for this astonishing advancement.
While there are many applications for face recognition in industries like security,
marketing, and user identification, it has also brought to light a serious privacy concern
due to the availability of the enormous datasets required to train these facial recognition
models. These databases frequently don't have strong encryption or security, making
them susceptible to hacks and unauthorized access. The privacy and security of
numerous people whose faces and identities are included in these databases might be
in danger if this vulnerability were to be exploited. A breach of face recognition
databases might result in identity theft as well as intrusive tracking, profiling, and
surveillance, which would fundamentally infringe on people's rights to privacy and
autonomy.
One solution would be to minimize the amount of data required for these machine
learning models, but it can be challenging to estimate the minimum amount of data you
require, especially with complex machine learning models like deep neural networks.
It is crucial to look for creative alternate solutions as a result of the seriousness of these
new privacy issues, particularly in apps that deal with users' privacy-sensitive
information including face data, electronic medical records, and location information.

#### 1.3 Privacy-Preserving Machine Learning (PPML)

We can broadly classify the machine learning pipeline into two phases– training and
serving.
There are traditionally four stages in an ML pipeline:

1. Data preprocessing
2. Training and evaluation
3. Deployment
4. Inference

The training phase mostly encompasses data collection, data preparation, training, and
evaluation of the model, while the serving phase deals with the usage of the model,
deploying the model, and inferring results.  
Privacy is a subjective term encompassing freedom of thought, control, seclusion of
one's body, information, and so on.
Since privacy is such a broad term with diverse points of view, it is difficult to define
or measure.
In a digital sense, at a bare minimum, privacy refers to the protection of one’s identity.
There exist several approaches to PPML, like implementing a full privacy-preserving
pipeline, or only privacy-preserving model generation and service.
Most currently existing or currently proposed privacy-preserving solutions stress that
implemented procedures must focus on the prevention of any sensitive data outflow,
especially that from the scope of training data sources.
There exist two broad methods of preserving privacy during model generation:
Processing training data in a privacy-preserving manner, and filtering the training data
to minimize any sensitive information.
Attempting to preserve privacy when you intend to share or publish information also
has a few proposed methods, which broadly fall into the following categories: methods
that require the complete abolition of any identifiable features in the raw data; methods that modify the probabilistic outcome of said raw data; or methods that entirely
augment the data using principles like confusion or diffusion.
The term "confusion-based approach" refers largely to a cryptographic technique that
uses confusion on the raw data in order to provide a substantially higher level of privacy
assurance than traditional anonymization techniques and perturbation-based
approaches.
With this approach, only the results of processing are meant to be publicized, and not
the actual data itself.

### CHAPTER 2: BACKGROUND & LITERATURE SURVEY

#### 2.1 Background

##### 2.1.1 History of Encryption

History is an aspect of our lives that stems from deep-rooted traditions of storytelling.
Before we had to means of recording any events, history was passed down through
several other mediums- like artwork, fables, etc. History itself is essentially a collection
of tales with lessons, even if the nature of the tales may be quite complex. The first
form of writing was largely pictorial, which despite being easy to understand, did not
allow for very complex records. As civilization began to become more complex, so did
their forms of writing- giving rise to some of the earliest written languages we know
today.
The concept of writing could be used not just for recording history, but also to store
information. The necessity of hiding the information and/or its meaning from prying
eyes was a concept that soon followed. The study of different methods of hiding the
contents of messages in any medium is known as Cryptography. Encryption refers to
the transformation of information into a hidden form to prevent malicious third parties
from accessing its contents, and this is merely one component of cryptography.
Around the year 2000 BC, we can see the first-ever evidence of the use of encryption-
the tomb of Khnumhotep II recorded the events of his life, however, the hieroglyphs
used to do so were not standard at the time. Education at the time was a privilege only
affordable to the elite, and while these hieroglyphs may have been used with the
intention of showing of one’s writing skills, or discussing taboo subjects, it also had an
unintended effect- the meaning of the original text was obscured, yet still recorded.
Around 500 BC, we can see the first recorded use of encryption in military
applications- the Spartans used an invention called the scytale, which allowed the
transmission of secret messages. The scytale was a cylindrical device along which a
paper was wound, and the message written. Without the use of a scytale, it would have
been both, impossible to read or write these messages. The scytale in this case was the ‘key’ 
used for both obscuring the message (encryption) and understanding its contents
(decryption)
One of the most important and well-known ciphers of all time was the ‘Caeser Cipher’,
used by the Roman legions under Julius Caesar. It was only cracked in 800 AD. The
Caeser Cipher makes use of an alphabet shift, where each letter is substituted with the
letter certain paces before/after it.

![image to be added]()

Despite the simplicity and effectiveness of the Caeser Cipher, its monoalphabetic 
rotation had one major drawback. The encryption was substituting each instance of a 
letter with the same letter, making it vulnerable to a frequency analysis, where we 
compare the occurrence of each letter in a given text, in a given language, and compare 
it to the expected frequency of each letter in that language. 
A very important encryption algorithm that is widely used in the present day is the RSA 
Algorithm (named after its creators Ron Rivsest, Adi Shamir, and Leonard Adleman). 
The algorithm uses the concept of asymmetric cryptography, where encryption and 
decryption are not done using the same key, but rather two different keys or sets of 
keys. They believed that key transmission was not necessary and that a message could 
be encrypted using an encryption key, which would be made publicly available by the 
intended recipient. The message would then subsequently be decrypted by that same 
person, and only they would possess the ability to decipher the message. The 
encryption algorithm works as follows: 
Consider e and n to be two positive integers. These will serve as our public key.
First, we would represent the message as an integer between 0 and n-1. In the event 
that the message is larger than n, we may choose to break it into blocks and represent 
each block as an integer between 0 and n-1. 
Then we would encrypt the message by raising it to the power of e and then dividing it 
by n. The resulting ciphertext C is the remainder of the division by n. 
C = Me % n. 
For decryption, we would use a pair of two positive integers, d and n, which will serve 
as our private key. 
To obtain the original plaintext M, we will raise the ciphertext C to the power of d, and 
divide this by n. The resulting remainder will be our plaintext M. 
M = Cd % n

##### 2.1.2 Understanding Different Image Encryption Methods 

Images refer to electronic files, which consist of one or more matrices of numbers, 
where each matrix represents a colour or transparency channel (one being grayscale, 
three being RGB, etc.). 
Each value in this matrix is called a pixel, representing a specific colour. 
A digital image I is usually defined as I (H×W×C), where H, W and C are the height, 
width and number of channels present within the image.  
Images usually contain loads of information, much of which is private or sensitive, like 
medical or military imaging, legal or personal documents, etc. 
With the increased traction of image sharing through social media platforms and share 
of images over the network, these images are vulnerable to being compromised. Also, 
digital images have a high level of redundancy of information. 
Image encryption methods aim to ensure the confidentiality of the image from 
unauthorized access by hiding the ‘plaintext’ image and producing a ciphertext that can 
only be seen by authorized users. 
The two main principles used by image encryption algorithms are confusion and 
diffusion, both of which aim to reduce the aforementioned information redundancy
(correlation between pixels). Changing the value of each pixel of the plaintext image 
by substituting it with another pixel is known as confusion. Changing the location of 
each pixel in the plaintext image is known as diffusion. Diffusion helps reduce the 
correlation between close-by pixels in the plaintext image. Although several image 
algorithms have been developed or proposed, most of them use a key or set of keys to 
control the level of confusion and diffusion to obscure the contents of the plaintext 
image. 
The most commonly used algorithms for image encryption are generally stream 
ciphers, where the image is converted into a binary bit sequence of 1s and 0s, and then 
traditional encryption methods are used to generate the cipher image. The most 
common of these encryption algorithms used for images is the Advanced Encryption 
Standard (AES), which has been widely used for security purposes. It is a symmetric 
key encryption algorithm with variable key lengths. Depending on the key size, it 
performs between ten and fourteen rounds of encryption. Each bit stream is segmented 
into chunks of 128 bits, and in each round, several operations are performed on a given 
chunk or block of bits. These operations are `SubBytes`, in which each byte is replaced 
with a corresponding byte from a fixed look-up table (LUT); `ShiftRows`, where the 
bytes in each row are shifted by a certain offset value; `MixColumns`, within which 
four bytes of each column are combined using a linear transformation, which is 
invertible; and finally, `AddRoundKey`, where the subkey is combined with the current 
state. The result of all these subsequent transformations is a new block of bits, which 
serves as the starting block for the next round of the algorithm. 

![image to be added]()

Other stream cipher algorithms include Trivium and ChaCha20, both of which use very 
large secret keys (80 bits, 256 bits) to generate even larger bits streams, up to 264 bits 
in size. This bitstream is then XORed with the plaintext image, producing a cipher 
image. 
Chaos-based encryption algorithms are also common, wherein rulesets are employed 
to generate a specific output in the ciphertext based on a particular occurrence in the 
plaintext sequence, for example, DNA encoding encryption converts the image into a 
binary bit stream, and then applies a particular ruleset on the bitstream to produce the 
ciphertext, which in this case is a DNA sequence. 
‘Meaningful encryption’ is also a technique that is becoming increasingly popular, 
which utilizes the concept of encryption in conjunction with the concept of 
steganography, to produce cipher images that appear to contain meaningful 
information, rather than a traditional noise-like distortion that is obtained when 
performing image encryption. Steganography, in the concept of cybersecurity, is the 
practice of using overt messages or carrier files to conceal sensitive information within 
them. 

#### 2.2 Literature Survey 

##### 2.2.1 Privacy-preserving machine learning: Methods, challenges and directions 

This study reviews and summarizes existing methodologies that aim to preserve 
privacy and proposes a triad-based model to understand and evaluate these solutions 
by decomposing their privacy-preserving function. This paper provides a 
comprehensive overview of the field of privacy-preserving machine learning (PPML). 
Machine learning as a field is always being continually utilized in several application 
domains. Typically, the performance of a model depends on a substantial number of 
computational resources, as well as data required for training. The demand for, and 
utilization of large amounts of data gives rise to significant privacy concerns owing to 
the potential of disclosing sensitive information. furthermore, the dynamic institutions 
that have regulated increasingly tight access to use sensitive data add substantial 
obstacles to fully profiting from the capabilities of ML for data-driven applications. 
Membership, attribute, property, and model inversion attacks can also affect trained 
ML models, which is why an approach that preserves privacy is absolutely necessary. 

##### 2.2.2 HOG feature extraction from encrypted images for privacy-preserving machine learning

The paper proposes a novel method for extracting Histogram of Oriented-Gradients 
(HOG) features from encrypted images to enable privacy-preserving machine learning 
[1]. In this study, the images are encrypted using a block-based encryption method 
followed by a novel block-based extraction method of HOG. The proposed extraction 
method includes dividing images into small grids (HOG cells), quantizing gradient 
angles, creating gradient histograms in cells, and forming HOG blocks by grouping 
HOG cells. To demonstrate its effectiveness, this approach is applied to a face image 
recognition problem, using YALE Extended Face Database under the use of two kinds 
of classifiers: linear support vector machine (SVM) and Gaussian SVM classifiers. The 
experimental results demonstrate that the encryption process has minimal impact on 
the performance of SVM algorithms under certain parameter conditions, highlighting 
the efficacy of the suggested method in maintaining accuracy while safeguarding the 
privacy of visual data. The findings suggest that the encryption procedure does not 
significantly impede the efficiency of machine learning algorithms, highlighting the 
potential for secure data processing without compromising classification accuracy. In 
summary, the study underlines the practicality of using encrypted photos for machine 
learning purposes, hence resolving privacy issues in data-driven jobs. 