.PHONY: macos

macos: clean app pkg

app:
	python3 -m PyInstaller -F --windowed macos.spec

pkg:
	./package-macos.sh

clean:
	rm -r dist/ build/
