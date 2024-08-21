import sys
from data_config import DataConfig

class Menu:
    def __init__(self, log_path):
        self.dc = DataConfig()

    def cli_menu(self):
        while True:
            print("""
        Simple CSV Grapher Menu:
        1.) Import Data from CSV File
        2.) Remove Data
        3.) Set XY Labels
        4.) Set Title
        5.) Choose Graph
        6.) Plot Data
        7.) Quit
        """)
            option = input("Option: ")
            if option == '1':
                filename = input('CSV Filename: ')
                headers = input('Give desired column headers (separated by spaces): ')
                col_names = headers.split(' ')
                self.dc.add_data(self.log_path, filename, col_names)
            elif option == '2':
                print("Current data:", list(self.data.keys()))
                file = input("Remove one of the following csv files: ")
                self.dc.remove_data(file)
            elif option == '3':
                x_label = input("X Label: ")
                y_label = input("Y Label: ")
                z_label = input("Z Lavel: ")
                self.dc.set_labels([x_label, y_label, z_label])
            elif option == '4':
                title = input('Set Title: ')
                self.dc.set_title(title)
            elif option == '5':
                type = input('Choose one of the following ["line", "scatter"]: ')
                self.dc.set_graph_type(type)
            elif option == '6':
                live_input = input("Do you want to live view data (y/n):")
                is_live_view = True if live_input == 'y' else False
                if is_live_view == False:
                    animated_input = input("Do you want to save an animated graph (y/n): ")
                    is_animated = True if animated_input == 'y' else False
                    if is_animated == False:
                        save_input = input("Save graph as file (y/n): ")
                        save_file = True if save_input == 'y' else False
                    else:
                        save_file = False
                else:
                    is_animated, save_file = False, False
                self.dc.set_live_view(is_live_view)
                self.dc.set_animation(is_animated)
                self.dc.can_save(save_file)

                can_plot = self.dc.can_plot()
                if can_plot == True:
                    break
                else:
                    print("The title, x and y labels and/or data has not been set yet.")
            elif option == '7':
                sys.exit(0)
            else:
                print("Invalid option: %s"%(option))