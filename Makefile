.PHONY: macos

macos:
	python3 -m PyInstaller -F --windowed ./build.spec

clean:
	rm -r dist/ build/
