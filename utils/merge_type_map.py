GTR_TYPE_MAP = {
    # Publications
    "journal article/review": ("publication", "article_or_review"),
    "conference/paper/proceeding/abstract": ("publication", "conference_proceedings_paper_abstract_or_review"),
    "preprint": ("publication", "preprint"),
    "working paper": ("publication", "working_paper"),
    "systematic review": ("publication", "systematic_review"),
    "monograph": ("publication", "monograph"),
    "manual/guide": ("publication", "manual"),
    "technical standard": ("publication", "standard"),
    "thesis": ("publication", "thesis"),
    "technical report": ("publication", "report"),
    "policy briefing/report": ("publication", "report"),
    "consultancy report": ("publication", "report"),
    "book chapter": ("publication", "book_chapter"),
    "book": ("publication", "book"),
    "book edited": ("publication", "edited_book"),
    "report": ("publication", "report"),
    "other": ("publication", "other"),

    # Databases or models
    "database/collection of data": ("database_or_model", "dataset"),
    "computer model/algorithm": ("database_or_model", "model_or_algorithm"),
    "data analysis technique": ("database_or_model", "data_analysis_technique"),

    # Software and technical products
    "software": ("software_and_technical_product", "software"),
    "webtool/application": ("software_and_technical_product", "web_application"),
    "e-business platform": ("software_and_technical_product", "ebusiness_platform"),
    "physical model/kit": ("software_and_technical_product", "physical_model_or_kit"),
    "new/improved technique/technology": ("software_and_technical_product", "technique_or_technology"),
    "new material/compound": ("software_and_technical_product", "material_or_compound"),
    "systems, materials & instrumental engineering": ("software_and_technical_product", "engineering"),

    # Disseminations
    "participation in an activity, workshop or similar":("dissemination", "workshop_or_activity"),
    "participation in an open day or visit at my research institution":("dissemination", "visit"),
    "a talk or presentation":("dissemination", "talk_or_presentation"),
    "a press release, press conference or response to a media enquiry/interview":("dissemination", "press_release_conference_or_response"),
    "a formal working group, expert panel or dialogue":("dissemination", "working_group_expert_panel_or_dialogue"),
    "a broadcast e.g. tv/radio/film/podcast (other than news/press)": ("dissemination", "broadcast"),
    "engagement focused website, blog or social media channel": ("dissemination", "website_blog_social_media"),
    "a magazine, newsletter or online publication": ("dissemination", "magazine_newsletter_or_online_publication"),
    "scientific meeting (conference/symposium etc.)": ("dissemination", "scientific_meeting"),

    # Artistic and Creative products
    "film/video/animation": ("creative_product", "film_video_or_animation"),
    "composition/score": ("creative_product", "composition"),
    "performance (music, dance, drama, etc)":("creative_product", "performance"),
    "artistic/creative exhibition":("creative_product", "exhibition"),
    "artefact (including digital)": ("artistic_and_creative_product", "artefact"),
    "artwork": ("artistic_and_creative_product", "artwork"),
    "image": ("artistic_and_creative_product", "image"),
    "creative writing": ("artistic_and_creative_product", "creative_writing"),

    # Policy influences
    "contribution to a national consultation/review":("policy_influence", "consultation_or_review_contribution"),
    "implementation circular/rapid advice/letter to e.g. ministry of health":("policy_influence", "circular_rapid_advice_or_letter"),
    "participation in a guidance/advisory committee": ("policy_influence", "advisory_committee"),
    "membership of a guideline committee": ("policy_influence", "guideline_committee"),
    "influenced training of practitioners or researchers": ("policy_influence", "training"),
    "contribution to new or improved professional practice": ("policy_influence", "professional_practice"),
    "citation in other policy documents": ("policy_influence", "policy_citation"),
    "citation in systematic reviews": ("policy_influence", "systematic_review_citation"),

    # Research materials
    "technology assay or reagent": ("research_material", "assay_or_reagent"),
    "improvements to research infrastructure": ("research_material", "research_infrastructure"),
    "physiological assessment or outcome measure": ("research_material", "assessment_outcome_measure"),
    "biological samples": ("research_material", "biological_samples"),
    "model of mechanisms or symptoms - in vitro": ("product", "in_vitro_model"),
    "model of mechanisms or symptoms - mammalian in vivo": ("product", "mammalian_in_vivo_model"),
    "model of mechanisms or symptoms - human": ("product", "human_model"),

    # Products
    "products with applications outside of medicine": ("product", "non_medical_product"),
    "therapeutic intervention - medical devices": ("product", "medical_device"),
    "therapeutic intervention - vaccines": ("product", "vaccine"),

    # Extra GtR overarching categories
    "furtherfundings": ("further_funding", "further_funding"),
    "collaborations": ("collaboration", "collaboration"),
    "intellectualproperties": ("intellectual_property", "intellectual_property"),
    "spinouts": ("spinout", "spinout")
}

SCOPUS_TYPE_MAP = {
    # Publications
    "article": ("publication", "article_or_review"),
    "review": ("publication", "article_or_review"),
    "editorial": ("publication", "editorial"),
    "letter": ("publication", "letter"),
    "note": ("publication", "note"),
    "short survey": ("publication", "short_survey"),
    "erratum": ("publication", "erratum"),
    "retracted": ("publication", "retracted_article"),
    "conference paper": ("publication", "conference_proceedings_paper_abstract_or_review"),
    "conference review": ("publication", "conference_proceedings_paper_abstract_or_review"),
    "report": ("publication", "report"),
    "book": ("publication", "book"),
    "book chapter": ("publication", "book_chapter"),
    "data paper": ("publication", "data_paper"),
    "abstract report": ("publication", "abstract_report"),

    # Disseminations
    "business article": ("dissemination", "business_article"),
    "press release": ("dissemination", "press_release_conference_or_response"),
}


OPENALEX_TYPE_MAP = {
    # Publications
    "article": ("publication", "article_or_review"),
    "review": ("publication", "article_or_review"),
    "preprint": ("publication", "preprint"),
    "editorial": ("publication", "editorial"),
    "letter": ("publication", "letter"),
    "erratum": ("publication", "erratum"),
    "peer-review": ("publication", "peer_review"),
    "conference-paper": ("publication", "conference_proceedings_paper_abstract_or_review"),
    "conference-abstract": ("publication", "conference_proceedings_paper_abstract_or_review"),
    "book": ("publication", "book"),
    "book-chapter": ("publication", "book_chapter"),
    "report": ("publication", "report"),
    "dissertation": ("publication", "thesis"),
    "standard": ("publication", "standard"),
    "data-paper": ("publication", "data_paper"),

    # Databases and models
    "dataset": ("database_or_model", "dataset"),

    # Other / unknown
    "retraction": ("other", "retraction"),
    "reference-entry": ("other", "reference_entry"),
    "paratext": ("other", "paratext"),
    "libguides": ("dissemination", "libguide"),
    "supplementary-materials": ("other", "supplementary_material"),
    "other": ("other", "other"),
}

WOS_TYPE_MAP = {
    # Publications
    "article": ("publication", "article_or_review"),
    "review": ("publication", "article_or_review"),
    "letter": ("publication", "letter"),
    "editorial material": ("publication", "editorial"),
    "proceedings paper": ("publication", "conference_proceedings_paper_abstract_or_review"),
    "meeting abstract": ("publication", "conference_proceedings_paper_abstract_or_review"),
    "book": ("publication", "book"),
    "article; proceedings paper": ("publication", "conference_proceedings_paper_abstract_or_review"),
    "review; book chapter": ("publication", "book_chapter"),
    "editorial material; book chapter": ("publication", "book_chapter"),
    "article; early access": ("publication", "article_or_review"),
    "review; early access": ("publication", "article_or_review"),
    "editorial material; early access": ("publication", "editorial"),
    "article; book chapter": ("publication", "book_chapter"),
    "article; retracted publication": ("publication", "retracted_article"),
    "retracted publication": ("publication", "retracted_article"),
    "withdrawn publication": ("publication", "retracted_article"),
    "publication with expression of concern": ("publication", "publication_with_expression_of_concern"),

    # Dissemination
    "news item": ("dissemination", "magazine_newsletter_or_online_publication"),

    # Other
    "correction": ("other", "correction"),
    "retraction": ("other", "retraction"),
    "expression of concern": ("other", "expression_of_concern"),
    "item withdrawal": ("other", "withdrawal"),
}