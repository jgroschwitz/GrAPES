"""
writes decrypted licensed data for unbounded dependencies and word disambiguation
"""
import penman

# use test_corpus for testing
main_corpus = "test_corpus"
# main_corpus = "corpus"
path_to_corpus_folder = "../../corpus/"


def write_ptb_data(decrypted_txt, decrypted_tsv):
    """
    Write the contents of the AMR and TSV files for descripted PTB sentences to files
    individual AMR corpus:
        corpus/subcorpora/unbounded_dependencies.txt
    append to corpus/corpus.txt
    TSV: corpus/unbounded_dependencies.tsv

    Args:
        decrypted_txt: the content of the AMR file
        decrypted_tsv: the content of the TSV file
    """
    subcorpus = "unbounded_dependencies"

    # check if we already have everything in place
    append = True
    create_txt = False
    create_tsv = False
    existing_main_corpus = penman.load(f"{path_to_corpus_folder}{main_corpus}.txt")
    sample_suppl = "cf21.mrg-or-1-0"
    found = [g for g in existing_main_corpus if g.metadata["suppl"] == sample_suppl]
    if len(found) > 0:
        print("looks like the entries are already in corpus.txt.")
        append = False
    try:
        f = open(f"{path_to_corpus_folder}/subcorpora/{subcorpus}.txt", 'r')
        f.close()
        print(f"looks like you've already got the corpus/subcorpora/{subcorpus}.txt file")
    except FileNotFoundError:
        create_txt = True

    try:
        f = open(f"{path_to_corpus_folder}/{subcorpus}.tsv", 'r')
        f.close()
        print(f"looks like you've already got the corpus/{subcorpus}.tsv file")
    except FileNotFoundError:
        create_tsv = True

    if not (append or create_txt or create_tsv):
        print("Nothing to do; exiting")
        return

    amrs = penman.loads(decrypted_txt)

    for i, entry in enumerate(amrs):
        # copy old ID to supplementary info
        entry.metadata["suppl"] = entry.metadata["id"]
        entry.metadata["id"] = f"{subcorpus}_{i}"

    # write to corpus files if needed
    if create_txt:
        penman.dump(amrs, f"{path_to_corpus_folder}/subcorpora/{subcorpus}.txt")
        print(f"added corpus/subcorpora/{subcorpus}.txt")
    if append:
        with open(f"{path_to_corpus_folder}/{main_corpus}.txt", 'a') as corpus_file:
            corpus_file.write("\n")  # need blank line in between
            penman.dump(amrs, corpus_file)
            print(f"extended corpus/{main_corpus}.txt")

    # write to TSV if needed
    if create_tsv:
        with open(f"{path_to_corpus_folder}/{subcorpus}.tsv", 'w') as tsv_file:
            tsv_file.write(decrypted_tsv)
            print(f"added corpus/{subcorpus}.tsv")


def update_from_amr_testset(path_to_testset):
    """
    Updates the word_disambiguation subcorpus with the sentences from the test set
    Args:
        path_to_testset: path to the AMR 3.0 testset concatenated file
    """
    subcorpus = "word_disambiguation"
    # check if we already have everything in place
    append = True
    create_txt = False
    existing_main_corpus = penman.load(f"{path_to_corpus_folder}/{main_corpus}.txt")
    sample_suppl = "PROXY_NYT_ENG_20081128_0005.6"
    found = [g for g in existing_main_corpus if g.metadata["suppl"] == sample_suppl]
    if len(found) > 0:
        print("looks like the entries are already in corpus.txt.")
        append = False
    try:
        f = open(f"{path_to_corpus_folder}/subcorpora/{subcorpus}.txt", 'r')
        f.close()
        print(f"looks like you've already got the corpus/subcorpora/{subcorpus}.txt file")
    except FileNotFoundError:
        create_txt = True

    if not (create_txt or append):
        print("Nothing to do; exiting")
        return

    test_set = penman.load(path_to_testset)
    with open(f"{path_to_corpus_folder}/{subcorpus}_clean.txt", 'r') as subcorpus_file:
        amrs = penman.load(subcorpus_file)
        for g in amrs:
            if g.metadata["snt"] == "(removed -- see documentation)":
                id = g.metadata["suppl"]
                sentence = [gr.metadata["snt"] for gr in test_set if gr.metadata["id"] == id][0]
                g.metadata["snt"] = sentence

    if create_txt:
        penman.dump(amrs, f"{path_to_corpus_folder}/subcorpora/{subcorpus}.txt")
        print(f"added corpus/subcorpora/{subcorpus}.txt")
    if append:
        with open(f"{path_to_corpus_folder}/{main_corpus}.txt", "a") as c:
            c.write("\n")  # need blank line in between
            penman.dump(amrs, c)
            print(f"extended corpus/{main_corpus}.txt")


if __name__ == "__main__":
    # TODO delete this script part
    # the strings to write to files
    txt = """# CCG unbounded dependencies, annotated by Meaghan and students, built from relatives_meaghan.tsv object-free-relatives_chris.tsv object-relative-null_chris.tsv object_wh_questions_chris.tsv right_node_raising_chris.tsv subj_relative_embedded_chris.tsv subj_relatives_chris.tsv 

# ::id cf21.mrg-or-1-0
# ::snt We have also developed techniques for recognizing and locating underground nuclear tests through the waves in the ground which they generate .
# ::cat object relative
# ::distance 6
(r / dummy)

# ::id cf03.mrg-or-3-1
# ::snt But these are dreamed in original action , in some particular continuity which we don't remember having seen in real life .
# ::cat object relative
# ::distance 7
(r / dummy)

# ::id cf22.mrg-or-1-2
# ::snt That finished the job that Captain Chandler and Lieutenant Carroll had begun .
# ::cat object relative
# ::distance 8
(r / dummy)

# ::id cf23.mrg-or-2-3
# ::snt There was the Hungarian Revolution which we praised and mourned , but did nothing about .
# ::cat object relative
# ::distance 4
(r / dummy)

# ::id cf23.mrg-or-2-4
# ::snt There was the Hungarian Revolution which we praised and mourned , but did nothing about .
# ::cat object relative
# ::distance 10
(r / dummy)

# ::id cg20.mrg-or-1-5
# ::snt Great stress is placed on the role that the monitoring of information sending plays in maintaining the effectiveness of the network .
# ::cat object relative
# ::distance 7
(r / dummy)

# ::id cg21.mrg-or-2-6
# ::snt This is important to understanding the position that doctrinaire liberals found themselves in after World War 2 , and our great democratic victory that brought no peace .
# ::cat object relative
# ::distance 4
(r / dummy)

# ::id ck02.mrg-or-2-7
# ::snt He could also hear the stream which he had seen from his position .
# ::cat object relative
# ::distance 4
(r / dummy)

# ::id cl01.mrg-or-1-8
# ::snt " There are things about me that I can't tell you now , Mary Jane " , I said , " but if you'll go out to dinner with me when I get out of Hanover , I'd like to tell you the whole story . "
# ::cat object relative
# ::distance 6
(r / dummy)

# ::id cl01.mrg-or-3-9
# ::snt It was a bad play , real grade-A turkey , which only a prevalence of angels with grandiose dreams of capital gain and tax money to burn could have put into rehearsal .
# ::cat object relative
# ::distance 20
(r / dummy)

# ::id cl02.mrg-or-1-10
# ::snt I took another sidelong glance at the other registration card , then took the key to Unit 13 that she had given me and went down long enough to park the car .
# ::cat object relative
# ::distance 7
(r / dummy)

# ::id cm03.mrg-or-2-11
# ::snt The simple treaty principle that Gabriel was asking him to ratify , in short , was nothing less than total trust .
# ::cat object relative
# ::distance 7
(r / dummy)

# ::id cp21.mrg-or-1-12
# ::snt Two days before he was taken sick , Cousin Elec was out worrying about what too much rain might do to his sweetpeas , and Cousin Elec had always preserved in the top drawer of his secretary a mother-of-pearl paper knife which Theresa had coveted as a child and which he had promised she could have when he died .
# ::cat object relative
# ::distance 4
(r / dummy)

# ::id cp21.mrg-or-1-13
# ::snt Two days before he was taken sick , Cousin Elec was out worrying about what too much rain might do to his sweetpeas , and Cousin Elec had always preserved in the top drawer of his secretary a mother-of-pearl paper knife which Theresa had coveted as a child and which he had promised she could have when he died .
# ::cat object relative
# ::distance 15
(r / dummy)

# ::id cf21.mrg-sr-2-14
# ::snt But when waves with a period of between 10 and 40 minutes begin to roll over the ocean , they set in motion a corresponding oscillation in a column of mercury which closes an electric circuit .
# ::cat subject relative
# ::distance 7
(r / dummy)

# ::id cf21.mrg-sr-4-15
# ::snt This center also receives prompt reports on earthquakes from four Coast Survey stations in the Pacific which are equipped with seismographs .
# ::cat subject relative
# ::distance 6
(r / dummy)

# ::id cl10.mrg-sr-3-16
# ::snt Indeed , there was a woman who , unasked , had already given him love .
# ::cat subject relative
# ::distance 5
(r / dummy)

# ::id cp10.mrg-sr-1-17
# ::snt There was the freshness of color , the freedom of perception , the lack of self-consciousness , but with a twist that made the forms leap from the page and smack you in the eye .
# ::cat subject relative
# ::distance 5
(r / dummy)

# ::id cp10.mrg-sr-1-18
# ::snt There was the freshness of color , the freedom of perception , the lack of self-consciousness , but with a twist that made the forms leap from the page and smack you in the eye .
# ::cat subject relative
# ::distance 10
(r / dummy)

# ::id cr07.mrg-sr-1-19
# ::snt I know something that is much more fun that we can do on our little lawn .
# ::cat subject relative
# ::distance 5
(r / dummy)

# ::id cr07.mrg-sr-1-20
# ::snt I know something that is much more fun that we can do on our little lawn .
# ::cat subject relative
# ::distance 9
(r / dummy)

# ::id cf21.mrg-ofr-1-0
# ::snt We have learned from earthquakes much of what we now know about the earth's interior , for they send waves through the earth which emerge with information about the materials through which they have traveled .
# ::cat object free relative
# ::distance 3
(r / dummy)

# ::id cf23.mrg-ofr-1-1
# ::snt The thing to lose sleep over is what people , having concluded that we are weaker than we are , are likely to do about it .
# ::cat object free relative
# ::distance 16
(r / dummy)

# ::id cf06.mrg-ofr-1-2
# ::snt Well , whatever you have , that 's it ! !
# ::cat object free relative
# ::distance 2
(r / dummy)

# ::id cf09.mrg-ofr-1-3
# ::snt Of course , her benevolence was limited to those who could afford it , but then there is a limit to what one person can do .
# ::cat object free relative
# ::distance 4
(r / dummy)

# ::id cg21.mrg-ofr-3-4
# ::snt What I am here to do is to report on the gyrations of the struggle -- a struggle that amounts to self-redefinition -- to see if we can predict its future course .
# ::cat object free relative
# ::distance 5
(r / dummy)

# ::id cg21.mrg-ofr-4-5
# ::snt Only Walter Lippman envisioned the possibility of our having " outlived most of what we used to regard as the program of our national purposes '' .
# ::cat object free relative
# ::distance 4
(r / dummy)

# ::id ck04.mrg-ofr-1-6
# ::snt The audience did not think much of the new pastor , and what the new pastor thought of the audience he did not dare at the time to say .
# ::cat object free relative
# ::distance 4
(r / dummy)

# ::id cf03.mrg-or0-2-0
# ::snt Skeptics may deny the more startling phenomena of dreams as things they have never personally observed , but failure to wonder at their basic mystery is outright avoidance of routine evidence .
# ::cat object relative with null complementizer
# ::distance 5
(r / dummy)

# ::id cf22.mrg-or0-1-1
# ::snt Undoubtedly none of the residents realize the influence their town has had on American military history , or the deeds of valor that have been done in its name .
# ::cat object relative with null complementizer
# ::distance 4
(r / dummy)

# ::id cf22.mrg-or0-2-2
# ::snt The truth is , however , that when Mel Chandler first reported to the regiment the only steed he had ever ridden was a swivel chair and the only weapon he had ever wielded was a pencil .
# ::cat object relative with null complementizer
# ::distance 4
(r / dummy)

# ::id cf22.mrg-or0-2-3
# ::snt The truth is , however , that when Mel Chandler first reported to the regiment the only steed he had ever ridden was a swivel chair and the only weapon he had ever wielded was a pencil .
# ::cat object relative with null complementizer
# ::distance 4
(r / dummy)

# ::id cf22.mrg-or0-4 -4
# ::snt There was no weapon Early could not take apart and reassemble blind-folded .
# ::cat object relative with null complementizer
# ::distance 4
(r / dummy)

# ::id cf22.mrg-or0-4 -5
# ::snt There was no weapon Early could not take apart and reassemble blind-folded .
# ::cat object relative with null complementizer
# ::distance 7
(r / dummy)

# ::id cg21.mrg-or0-1-6
# ::snt One of the obvious conclusions we can make on the basis of the last election , I suppose , is that we , the majority , were dissatisfied with Eisenhower conservatism .
# ::cat object relative with null complementizer
# ::distance 3
(r / dummy)

# ::id cg22.mrg-or0-1-7
# ::snt Have not our physical abilities already deteriorated because of the more sedentary lives we are now living ? ?
# ::cat object relative with null complementizer
# ::distance 4
(r / dummy)

# ::id ck10.mrg-or0-2 -8
# ::snt He is throttling the liberty my father gave his life to win ! !
# ::cat object relative with null complementizer
# ::distance 7
(r / dummy)

# ::id ck01.mrg-or0-1-9
# ::snt It required an energy he no longer possessed to be satirical about his father .
# ::cat object relative with null complementizer
# ::distance 4
(r / dummy)

# ::id xx00.mrg-oq-2-0
# ::snt What do the initials CPR stand for ?
# ::cat object wh-question
# ::distance 5
(r / dummy)

# ::id xx00.mrg-oq-5-1
# ::snt What does target heart rate mean ?
# ::cat object wh-question
# ::distance 5
(r / dummy)

# ::id xx00.mrg-oq-6-2
# ::snt What is the text of an opera called ?
# ::cat object wh-question
# ::distance 7
(r / dummy)

# ::id xx00.mrg-oq-7-3
# ::snt What did Peter Minuit buy for the equivalent of $ 24.00 ?
# ::cat object wh-question
# ::distance 4
(r / dummy)

# ::id xx00.mrg-oq-11-4
# ::snt How much vitamin C should you take in a day ?
# ::cat object wh-question
# ::distance 3
(r / dummy)

# ::id xx00.mrg-oq-12-5
# ::snt What did Edward Binney and Howard Smith invent in 1903 ?
# ::cat object wh-question
# ::distance 7
(r / dummy)

# ::id xx00.mrg-oq-17-6
# ::snt What state does Martha Stewart live in ?
# ::cat object wh-question
# ::distance 4
(r / dummy)

# ::id xx00.mrg-oq-19-7
# ::snt What countries has the IFC financed projects in ?
# ::cat object wh-question
# ::distance 4
(r / dummy)

# ::id cf21.mrg-rnr-1-0
# ::snt The waves of a 1923 tsunami in Sagami Bay brought to the surface and battered to death huge numbers of fishes that normally live at a depth of 3,000 feet .
# ::cat right node raising
# ::distance 9
(r / dummy)

# ::id cf21.mrg-rnr-1-1
# ::snt The waves of a 1923 tsunami in Sagami Bay brought to the surface and battered to death huge numbers of fishes that normally live at a depth of 3,000 feet .
# ::cat right node raising
# ::distance 4
(r / dummy)

# ::id cf21.mrg-rnr-2-2
# ::snt By means of charts showing wave-travel times and depths in the ocean at various locations , it is possible to estimate the rate of approach and probable time of arrival at Hawaii of a tsunami getting under way at any spot in the Pacific .
# ::cat right node raising
# ::distance 12
(r / dummy)

# ::id cf21.mrg-rnr-2-3
# ::snt By means of charts showing wave-travel times and depths in the ocean at various locations , it is possible to estimate the rate of approach and probable time of arrival at Hawaii of a tsunami getting under way at any spot in the Pacific .
# ::cat right node raising
# ::distance 7
(r / dummy)

# ::id cf23.mrg-rnr-2-4
# ::snt The first was that America had become -- or was in danger of becoming -- a second-rate military power .
# ::cat right node raising
# ::distance 12
(r / dummy)

# ::id cf23.mrg-rnr-2-5
# ::snt The first was that America had become -- or was in danger of becoming -- a second-rate military power .
# ::cat right node raising
# ::distance 5
(r / dummy)

# ::id cf07.mrg-rnr-1-6
# ::snt Various factors in the setting can still be of great advantage in making the first intercourse a good rather than a bad memory for one or both .
# ::cat right node raising
# ::distance 5
(r / dummy)

# ::id cf07.mrg-rnr-1-7
# ::snt Various factors in the setting can still be of great advantage in making the first intercourse a good rather than a bad memory for one or both .
# ::cat right node raising
# ::distance 1
(r / dummy)

# ::id cf23.mrg-se-1-0
# ::snt There was the revolution in Tibet which we pretended did not exist .
# ::cat subject relative, embedded
# ::distance 8
(r / dummy)

# ::id cf10.mrg-se-1-1
# ::snt He then sold her some capsules that he asserted would take care of the tumors and cysts until she could collect the money for buying his machine .
# ::cat subject relative, embedded
# ::distance 5
(r / dummy)

# ::id cf30.mrg-se-1-2
# ::snt Today it espouses certain ideas in its curriculum that other institutions might consider somewhat breathtaking .
# ::cat subject relative, embedded
# ::distance 10
(r / dummy)

# ::id cg10.mrg-se-1-3
# ::snt They differed in the balance they believed essential to the sovereignty of the citizen -- but the supreme sacrifice each made served to maintain a still more fundamental truth : That individual life , liberty and happiness depend on a right balance between the two -- and on the limitation of sovereignty , in all its aspects , which this involves .
# ::cat subject relative, embedded
# ::distance 3
(r / dummy)

# ::id ck03.mrg-se-1-4
# ::snt Human nature was not a piece of meat you could tell was bad by its smell .
# ::cat subject relative, embedded
# ::distance 7
(r / dummy)

# ::id ck27.mrg-se-1-5
# ::snt `` That was the fille de chambre , the one you thought could n't get the eggs out .
# ::cat subject relative, embedded
# ::distance 2
(r / dummy)

# ::id cn04.mrg-se-2-6
# ::snt Next to him was a young boy I was sure had sat near me at one of the trading sessions .
# ::cat subject relative, embedded
# ::distance 5
(r / dummy)

# ::id wsj_0060.mrg-sr-3-0
# ::snt Viacom's move comes as the syndication market is being flooded with situation comedies that are still running on the networks .
# ::cat subject relative
# ::distance 4
(r / dummy)

# ::id wsj_0062.mrg-sr-3-1
# ::snt The first two issues featured ads from only a handful of big advertisers , including General Electric and Adolph Coors , but the majority were from companies like Waste Management Inc. and Bumkins International , firms that don't spend much money advertising and can't be relied on to support a magazine over the long haul .
# ::cat subject relative
# ::distance 3
(r / dummy)

# ::id wsj_0062.mrg-sr-3-2
# ::snt The first two issues featured ads from only a handful of big advertisers , including General Electric and Adolph Coors , but the majority were from companies like Waste Management Inc. and Bumkins International , firms that don't spend much money advertising and can't be relied on to support a magazine over the long haul .
# ::cat subject relative
# ::distance 10
(r / dummy)

# ::id wsj_0062.mrg-sr-4-3
# ::snt Billings weren't disclosed for the small account , which had been serviced at Young & Rubicam , New York .
# ::cat subject relative
# ::distance 5
(r / dummy)

# ::id wsj_0064.mrg-sr-1-4
# ::snt In September , the department had said it will require trucks and minivans to be equipped with the same front-seat headrests that have long been required on passenger cars .
# ::cat subject relative
# ::distance 5
(r / dummy)
    """
    tsv = """ID	sentence	source	edge	target	distance	category	comments
cf21.mrg-or-1-0	We have also developed techniques for recognizing and locating underground nuclear tests through the waves in the ground which they generate .	(g / generate-01)	:ARG1	(w / wave)	6	object relative	
cf03.mrg-or-3-1	But these are dreamed in original action , in some particular continuity which we don't remember having seen in real life .	(s / see-01)	:ARG1	(c / continuity)	7	object relative	
cf22.mrg-or-1-2	That finished the job that Captain Chandler and Lieutenant Carroll had begun .	(b / begin-01)	:ARG1	(j / job)	8	object relative	
cf23.mrg-or-2-3	There was the Hungarian Revolution which we praised and mourned , but did nothing about .	(a / and :op1 (p / praise-01) :op2 (m / mourn-01))	:ARG1	(e / event)	4	object relative	or (e / event  :name (n / name :op1 "Hungarian" :op2 "Revolution")) "
cf23.mrg-or-2-4	There was the Hungarian Revolution which we praised and mourned , but did nothing about .	(d / do-02)	:ARG2	(e / event) 	10	object relative	or (e / event  :name (n / name :op1 "Hungarian" :op2 "Revolution")) 
cg20.mrg-or-1-5	Great stress is placed on the role that the monitoring of information sending plays in maintaining the effectiveness of the network .	(p / play-02)	:ARG1	(r / role)	7	object relative	
cg21.mrg-or-2-6	This is important to understanding the position that doctrinaire liberals found themselves in after World War 2 , and our great democratic victory that brought no peace .	(f / find-01)	:ARG1	(p / position-01)	4	object relative	or 6 (to ’n’)
ck02.mrg-or-2-7	He could also hear the stream which he had seen from his position .	(s / see-01)	:ARG1	(s / stream)	4	object relative	or is it stream-thing?
cl01.mrg-or-1-8	" There are things about me that I can't tell you now , Mary Jane " , I said , " but if you'll go out to dinner with me when I get out of Hanover , I'd like to tell you the whole story . "	(t / tell-01)	:ARG1	(t2 / thing)	6	object relative	
cl01.mrg-or-3-9	It was a bad play , real grade-A turkey , which only a prevalence of angels with grandiose dreams of capital gain and tax money to burn could have put into rehearsal .	(p / put-01) (r / rehearse-01)	:ARG1	(p / play-11) (t / turkey)	20	object relative	up to 25
cl02.mrg-or-1-10	I took another sidelong glance at the other registration card , then took the key to Unit 13 that she had given me and went down long enough to park the car .	(g / give-01)	:ARG1	(k / key)	7	object relative	
cm03.mrg-or-2-11	The simple treaty principle that Gabriel was asking him to ratify , in short , was nothing less than total trust .	(r / ratify-01)	:ARG1	(p / principle)	7	object relative	
cp21.mrg-or-1-12	Two days before he was taken sick , Cousin Elec was out worrying about what too much rain might do to his sweetpeas , and Cousin Elec had always preserved in the top drawer of his secretary a mother-of-pearl paper knife which Theresa had coveted as a child and which he had promised she could have when he died .	(c / covet-01)	:ARG1	(k / knife)	4	object relative	
cp21.mrg-or-1-13	Two days before he was taken sick , Cousin Elec was out worrying about what too much rain might do to his sweetpeas , and Cousin Elec had always preserved in the top drawer of his secretary a mother-of-pearl paper knife which Theresa had coveted as a child and which he had promised she could have when he died .	(h / have-03)	:ARG1	(k / knife)	15	object relative	
cf21.mrg-sr-2-14	But when waves with a period of between 10 and 40 minutes begin to roll over the ocean , they set in motion a corresponding oscillation in a column of mercury which closes an electric circuit .	(c / close-01)	:ARG0	(o / oscillate-01)	7	subject relative	
cf21.mrg-sr-4-15	This center also receives prompt reports on earthquakes from four Coast Survey stations in the Pacific which are equipped with seismographs .	(e / equip-01)	:ARG1	(s / station)	6	subject relative	
cl10.mrg-sr-3-16	Indeed , there was a woman who , unasked , had already given him love .	(g / give-01) (l / love-01)	:ARG0	(w / woman)	5	subject relative	
cp10.mrg-sr-1-17	There was the freshness of color , the freedom of perception , the lack of self-consciousness , but with a twist that made the forms leap from the page and smack you in the eye .	(t / twist-01) (t / twist-01 :ARG1 (t2 / thing))	:ARG0	(l / leap-03)	5	subject relative	
cp10.mrg-sr-1-18	There was the freshness of color , the freedom of perception , the lack of self-consciousness , but with a twist that made the forms leap from the page and smack you in the eye .	(t / twist-01) (t / twist-01 :ARG1 (t2 / thing))	:ARG0	(s / smack-02)	10	subject relative	
cr07.mrg-sr-1-19	I know something that is much more fun that we can do on our little lawn .	(f / fun-01)	:ARG1	(s / something)	5	subject relative	
cr07.mrg-sr-1-20	I know something that is much more fun that we can do on our little lawn .	(d / do-02)	:ARG0	(s / something)	9	subject relative	
cf21.mrg-ofr-1-0	We have learned from earthquakes much of what we now know about the earth's interior , for they send waves through the earth which emerge with information about the materials through which they have traveled .	(k / know-01)	:ARG1	(t / thing)	3	object free relative	
cf23.mrg-ofr-1-1	The thing to lose sleep over is what people , having concluded that we are weaker than we are , are likely to do about it .	(d / do-02)	:ARG1	(t / thing)	16	object free relative	
cf06.mrg-ofr-1-2	Well , whatever you have , that 's it ! !	(h / have-03)	:ARG1	(w / whatever)	2	object free relative	
cf09.mrg-ofr-1-3	Of course , her benevolence was limited to those who could afford it , but then there is a limit to what one person can do .	(d / do-02)	:ARG1	(t / thing)	4	object free relative	
cg21.mrg-ofr-3-4	What I am here to do is to report on the gyrations of the struggle -- a struggle that amounts to self-redefinition -- to see if we can predict its future course .	(d / do-02)	:ARG1	(t / thing)	5	object free relative	
cg21.mrg-ofr-4-5	Only Walter Lippman envisioned the possibility of our having " outlived most of what we used to regard as the program of our national purposes " .	(r / regard-01)	:ARG1	(t / thing)	4	object free relative	
ck04.mrg-ofr-1-6	The audience did not think much of the new pastor , and what the new pastor thought of the audience he did not dare at the time to say .	(t / think-01)	:ARG1	(t / thing)	4	object free relative	
cf03.mrg-or0-2-0	Skeptics may deny the more startling phenomena of dreams as things they have never personally observed , but failure to wonder at their basic mystery is outright avoidance of routine evidence .	(o / observe-01)	:ARG1	(t / thing)	5	object relative with null complementizer	
cf22.mrg-or0-1-1	Undoubtedly none of the residents realize the influence their town has had on American military history , or the deeds of valor that have been done in its name .	(h / have-03)	:ARG1	(i / influence-01)	4	object relative with null complementizer	
cf22.mrg-or0-2-2	The truth is , however , that when Mel Chandler first reported to the regiment the only steed he had ever ridden was a swivel chair and the only weapon he had ever wielded was a pencil .	(r / ride-01)	:ARG1	(s2 / steed)	4	object relative with null complementizer	
cf22.mrg-or0-2-3	The truth is , however , that when Mel Chandler first reported to the regiment the only steed he had ever ridden was a swivel chair and the only weapon he had ever wielded was a pencil .	(w / wield-01)	:ARG1	(w2 / weapon)	4	object relative with null complementizer	
cf22.mrg-or0-4 -4	There was no weapon Early could not take apart and reassemble blind-folded .	(d / disassemble-01)	:ARG1	(w / weapon)	4	object relative with null complementizer	No 1:1 frame for "take apart" - chose "disassemble-01"
cf22.mrg-or0-4 -5	There was no weapon Early could not take apart and reassemble blind-folded .	(d / reassemble-01)	:ARG1	(w / weapon)	7	object relative with null complementizer	
cg21.mrg-or0-1-6	One of the obvious conclusions we can make on the basis of the last election , I suppose , is that we , the majority , were dissatisfied with Eisenhower conservatism .	(m / make-01)	:ARG1	(c / conclusion)	3	object relative with null complementizer	
cg22.mrg-or0-1-7	Have not our physical abilities already deteriorated because of the more sedentary lives we are now living ? ?	(l / live-01)	:ARG1	(l2 / life)	4	object relative with null complementizer	
ck10.mrg-or0-2 -8	He is throttling the liberty my father gave his life to win ! !	(w / win-01)	:ARG5	(l / liberty)	7	object relative with null complementizer	
ck01.mrg-or0-1-9	It required an energy he no longer possessed to be satirical about his father .	(p / possess-01)	:ARG1	(e / energy)	4	object relative with null complementizer	
xx00.mrg-oq-2-0	What do the initials CPR stand for ?	(s / stand-08)	:ARG1	(a / amr-unkown)	5	object wh-question	
xx00.mrg-oq-5-1	What does target heart rate mean ?	(m / mean-01)	:ARG1	(a / amr-unkown)	5	object wh-question	
xx00.mrg-oq-6-2	What is the text of an opera called ?	(c / call-01)	:ARG1	(a / amr-unkown)	7	object wh-question	
xx00.mrg-oq-7-3	What did Peter Minuit buy for the equivalent of $ 24.00 ?	(b / buy-01)	:ARG1	(a / amr-unkown)	4	object wh-question	
xx00.mrg-oq-11-4	How much vitamin C should you take in a day ?	(t / take-01)	:ARG1	(s / small-molecule :name (n / name :op1 "vitamin" :op2 "C"))	3	object wh-question	
xx00.mrg-oq-12-5	What did Edward Binney and Howard Smith invent in 1903 ?	(i / invent-01)	:ARG1	(a / amr-unkown)	7	object wh-question	
xx00.mrg-oq-17-6	What state does Martha Stewart live in ?	(l / live-01) 	:ARG1	(s / state)	4	object wh-question	
xx00.mrg-oq-19-7	What countries has the IFC financed projects in ?	(f / finance-01)	:ARG1	(c / country)	4	object wh-question	
cf21.mrg-rnr-1-0	The waves of a 1923 tsunami in Sagami Bay brought to the surface and battered to death huge numbers of fishes that normally live at a depth of 3,000 feet .	(b / bring-01)	:ARG1	(n / number)	9	right node raising	
cf21.mrg-rnr-1-1	The waves of a 1923 tsunami in Sagami Bay brought to the surface and battered to death huge numbers of fishes that normally live at a depth of 3,000 feet .	(b / batter-01)	:ARG1	(n / number)	4	right node raising	
cf21.mrg-rnr-2-2	By means of charts showing wave-travel times and depths in the ocean at various locations , it is possible to estimate the rate of approach and probable time of arrival at Hawaii of a tsunami getting under way at any spot in the Pacific .	(r / rate-01)	:ARG1	(t / tsunami)	12	right node raising	
cf21.mrg-rnr-2-3	By means of charts showing wave-travel times and depths in the ocean at various locations , it is possible to estimate the rate of approach and probable time of arrival at Hawaii of a tsunami getting under way at any spot in the Pacific .	(t / tsunami)	:time	(a / arrive-01)	7	right node raising	
cf23.mrg-rnr-2-4	The first was that America had become -- or was in danger of becoming -- a second-rate military power .	(b / become-01)	:ARG2	(p / power)	12	right node raising	
cf23.mrg-rnr-2-5	The first was that America had become -- or was in danger of becoming -- a second-rate military power .	(b / become-01)	:ARG2	(p / power)	5	right node raising	
cf07.mrg-rnr-1-6	Various factors in the setting can still be of great advantage in making the first intercourse a good rather than a bad memory for one or both .	(g / good-02)	:ARG1	(m / memory)	5	right node raising	
cf07.mrg-rnr-1-7	Various factors in the setting can still be of great advantage in making the first intercourse a good rather than a bad memory for one or both .	(b / bad-07)	:ARG1	(m / memory)	1	right node raising	
cf23.mrg-se-1-0	There was the revolution in Tibet which we pretended did not exist .	(e / exist-01)	:ARG1	(r / revolution)	8	subject relative, embedded	
cf10.mrg-se-1-1	He then sold her some capsules that he asserted would take care of the tumors and cysts until she could collect the money for buying his machine .	(c / care-03)	:ARG1	(c2 / capsule)	5	subject relative, embedded	
cf30.mrg-se-1-2	Today it espouses certain ideas in its curriculum that other institutions might consider somewhat breathtaking .	(s / stun-01)	:ARG0	(i / idea)	10	subject relative, embedded	No 1:1 frame for breathtaking - chose "stun-01"
cg10.mrg-se-1-3	They differed in the balance they believed essential to the sovereignty of the citizen -- but the supreme sacrifice each made served to maintain a still more fundamental truth : That individual life , liberty and happiness depend on a right balance between the two -- and on the limitation of sovereignty , in all its aspects , which this involves .	(e / essential-01)	:ARG1	(b / balance)	3	subject relative, embedded	
ck03.mrg-se-1-4	Human nature was not a piece of meat you could tell was bad by its smell .	(b / bad-07)	:ARG1	(p / piece)	7	subject relative, embedded	
ck27.mrg-se-1-5	`` That was the fille de chambre , the one you thought could n't get the eggs out .	(g / get-05)	:ARG1	(o / one)	2	subject relative, embedded	
cn04.mrg-se-2-6	Next to him was a young boy I was sure had sat near me at one of the trading sessions .	(s / sit-01)	:ARG1	(b / boy)	5	subject relative, embedded	
wsj_0060.mrg-sr-3-0	Viacom's move comes as the syndication market is being flooded with situation comedies that are still running on the networks .	(r / run-09)	:ARG1	(c / comedy)	4	subject relative	
wsj_0062.mrg-sr-3-1	The first two issues featured ads from only a handful of big advertisers , including General Electric and Adolph Coors , but the majority were from companies like Waste Management Inc. and Bumkins International , firms that don't spend much money advertising and can't be relied on to support a magazine over the long haul .	(s / spend-01)	:ARG0	(f / firm)	3	subject relative	
wsj_0062.mrg-sr-3-2	The first two issues featured ads from only a handful of big advertisers , including General Electric and Adolph Coors , but the majority were from companies like Waste Management Inc. and Bumkins International , firms that don't spend much money advertising and can't be relied on to support a magazine over the long haul .	(r / rely-01)	:ARG0	(f / firm)	10	subject relative	
wsj_0062.mrg-sr-4-3	Billings weren't disclosed for the small account , which had been serviced at Young & Rubicam , New York .	(s / service-05)	:ARG1	(a / account)	5	subject relative	
wsj_0064.mrg-sr-1-4	In September , the department had said it will require trucks and minivans to be equipped with the same front-seat headrests that have long been required on passenger cars .	(r / require-01)	:ARG1	(h / headrest)	5	subject relative	
"""

    write_ptb_data(txt, tsv)

    # update_from_amr_testset("/home/meaghan/datasets/AMR 3.0/test.txt")
