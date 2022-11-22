
class FileGenerator:
    def Generate_Annotation(this, txt_path, line):
        with open(txt_path + '.txt', 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')