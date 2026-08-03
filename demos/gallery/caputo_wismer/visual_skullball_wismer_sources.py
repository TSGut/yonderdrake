"""Three Caputo-Wismer sources aimed through the stylized skullball."""

try:
    from ._visual_skullball_wismer import main
except ImportError:
    from _visual_skullball_wismer import main

if __name__ == "__main__":
    main("sources")
