"""Lower-bound baseline: every word is its own lemma."""


class IdentityLemmatizer:
    def lemmatize(self, words: list[str]) -> list[str]:
        return list(words)
