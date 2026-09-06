from nmd_host.intelligence.rules import RuleBasedExtractor
from nmd_host.intelligence.schemas import Conversation

extractor = RuleBasedExtractor()
conversation = Conversation.from_user_message("My name is Sathya. I work with Python.")
decisions = extractor.extract(conversation)
for d in decisions:
    print(f"Category: {d.candidate.category}, Text: {d.candidate.text}, Should remember: {d.should_remember}")