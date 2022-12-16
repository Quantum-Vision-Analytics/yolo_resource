class FileGenerator:
    # Save annotation files
    def Generate_Annotation(this, txt_path, line):
        with open(txt_path + '.txt', 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')

    # Write all the class names on a file
    def Generate_Classes(this, dir, names):
        with open(dir + '/classes.txt', 'w') as clsFile:
                for cls in names:
                    clsFile.write(cls + "\n")