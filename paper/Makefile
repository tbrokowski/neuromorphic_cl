# Makefile for Neuromorphic Continual Learning Paper

# Paper name (without extension)
PAPER = neuromorphic_continual_learning

# LaTeX compiler
LATEX = pdflatex
BIBTEX = bibtex

# Compilation flags
LATEX_FLAGS = -interaction=nonstopmode -file-line-error

# Default target
all: $(PAPER).pdf

# Compile PDF
$(PAPER).pdf: $(PAPER).tex references.bib
	$(LATEX) $(LATEX_FLAGS) $(PAPER).tex
	$(BIBTEX) $(PAPER)
	$(LATEX) $(LATEX_FLAGS) $(PAPER).tex
	$(LATEX) $(LATEX_FLAGS) $(PAPER).tex

# Quick compile (no bibliography)
quick: $(PAPER).tex
	$(LATEX) $(LATEX_FLAGS) $(PAPER).tex

# Clean auxiliary files
clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot
	rm -f *.fdb_latexmk *.fls *.synctex.gz
	rm -f *.nav *.snm *.vrb
	rm -f $(PAPER).pdf

# Clean everything including PDF
distclean: clean
	rm -f $(PAPER).pdf

# Count words (approximate)
wordcount:
	@echo "Approximate word count:"
	@texcount -brief $(PAPER).tex

# Check for common LaTeX issues
check:
	@echo "Checking for common issues..."
	@grep -n "\\\\cite{}" $(PAPER).tex || echo "No empty citations found"
	@grep -n "\\\\ref{}" $(PAPER).tex || echo "No empty references found"  
	@grep -n "PLACEHOLDER" $(PAPER).tex && echo "Warning: Placeholders still present!" || echo "No placeholders found"

# View PDF (Linux)
view: $(PAPER).pdf
	xdg-open $(PAPER).pdf

# View PDF (macOS)
view-mac: $(PAPER).pdf
	open $(PAPER).pdf

# Spell check (requires aspell)
spell:
	aspell check $(PAPER).tex

# Generate anonymous version for review
anonymous: $(PAPER).tex
	sed 's/Anonymous Authors/\\author{Anonymous}/g' $(PAPER).tex > $(PAPER)_anonymous.tex
	sed -i 's/Anonymous Institution//g' $(PAPER)_anonymous.tex
	sed -i 's/anonymous@institution.edu//g' $(PAPER)_anonymous.tex
	$(LATEX) $(LATEX_FLAGS) $(PAPER)_anonymous.tex
	$(BIBTEX) $(PAPER)_anonymous
	$(LATEX) $(LATEX_FLAGS) $(PAPER)_anonymous.tex
	$(LATEX) $(LATEX_FLAGS) $(PAPER)_anonymous.tex

# Help target
help:
	@echo "Available targets:"
	@echo "  all        - Compile the paper with bibliography"
	@echo "  quick      - Quick compile without bibliography"
	@echo "  clean      - Remove auxiliary files"
	@echo "  distclean  - Remove all generated files including PDF"
	@echo "  wordcount  - Count approximate words"
	@echo "  check      - Check for common issues"
	@echo "  view       - Open PDF (Linux)"
	@echo "  view-mac   - Open PDF (macOS)"
	@echo "  spell      - Spell check (requires aspell)"
	@echo "  anonymous  - Generate anonymous version for review"
	@echo "  help       - Show this help message"

.PHONY: all quick clean distclean wordcount check view view-mac spell anonymous help
