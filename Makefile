# The reproduce targets, from the repository root: `make test` is `cd reproduce && make test` (and so for the others).
.PHONY: test numbers numbers-gpu figures check all
test numbers numbers-gpu figures check all:
	$(MAKE) -C reproduce $@
