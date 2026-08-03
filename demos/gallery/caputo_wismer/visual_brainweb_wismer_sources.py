"""Three Caputo-Wismer sources aimed through a BrainWeb head slice."""

try:
    from ._visual_brainweb_wismer import main
except ImportError:
    from _visual_brainweb_wismer import main

if __name__ == "__main__":
    main("sources")
